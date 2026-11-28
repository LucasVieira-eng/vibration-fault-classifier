"""Comparação controlada dos conjuntos temporais e espectrais para generalização."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from threadpoolctl import threadpool_limits

from src import hist_gradient_boosting, logistic_regression, random_forest
from src.config import FEATURES_CSV, PROCESSED_DIR, PROJECT_ROOT, TARGET_COLUMN
from src.evaluate import to_shape_features


SPECTRAL_FEATURES_CSV = PROCESSED_DIR / "features_with_spectral.csv"
DOCS_DIR = PROJECT_ROOT / "docs"
MODELS = {
    "Random Forest": random_forest.make_model,
    "Hist Gradient Boosting": hist_gradient_boosting.make_model,
    "Logistic Regression": logistic_regression.make_model,
}
LABELS = [
    "Baseline",
    "Bearing",
    "Bearing+Unbalance",
    "Misalignment_NDE",
    "Unbalance_DE",
    "Unbalance_NDE",
]
BANDS = ["0_100hz", "100_500hz", "500_1500hz", "1500_3000hz"]
ACCELERATION_CHANNELS = ["acc_de_h", "acc_de_v", "acc_nde_h", "acc_nde_v"]


def load_data(path: str | Path = SPECTRAL_FEATURES_CSV):
    data = pd.read_csv(path)
    required = {TARGET_COLUMN, "rotation_hz", "file"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Colunas ausentes na tabela de features: {sorted(missing)}")
    if data["file"].duplicated().any():
        raise ValueError("A coluna file precisa identificar uma amostra sem duplicatas.")

    y = data[TARGET_COLUMN]
    groups = data["rotation_hz"]
    numeric = data.select_dtypes(include="number").drop(
        columns=["rotation_hz"], errors="ignore"
    )
    temporal_cols = [c for c in numeric if "_fft_rms_" not in c]
    spectral_cols = [c for c in numeric if c.startswith("acc_") and "_fft_rms_" in c]
    if len(temporal_cols) != 64:
        raise ValueError(f"Esperava 64 features temporais, encontrei {len(temporal_cols)}.")
    if len(spectral_cols) != 16:
        raise ValueError(f"Esperava 16 features espectrais de aceleração, encontrei {len(spectral_cols)}.")

    temporal = numeric[temporal_cols]
    shape = to_shape_features(temporal)
    reduced_cols = [
        c for c in temporal
        if c.endswith(("_rms", "_peak_to_peak", "_skewness", "_kurtosis"))
    ]
    if len(reduced_cols) != 32:
        raise ValueError(f"Esperava 32 features reduzidas, encontrei {len(reduced_cols)}.")

    spectral = numeric[spectral_cols]
    ilr_parts = []
    for channel in ACCELERATION_CHANNELS:
        columns = [f"{channel}_fft_rms_{band}" for band in BANDS]
        power = spectral[columns].pow(2).clip(lower=1e-24)
        ilr_parts.append(pd.DataFrame({
            f"{channel}_ilr_low_vs_lowmid": np.sqrt(1 / 2)
            * np.log(power[columns[0]] / power[columns[1]]),
            f"{channel}_ilr_lowmid_vs_mid": np.sqrt(2 / 3)
            * np.log(np.sqrt(power[columns[0]] * power[columns[1]]) / power[columns[2]]),
            f"{channel}_ilr_lowmid_vs_high": np.sqrt(3 / 4)
            * np.log((power[columns[0]] * power[columns[1]] * power[columns[2]]) ** (1 / 3)
                     / power[columns[3]]),
        }, index=spectral.index))
    ilr_spectral = pd.concat(ilr_parts, axis=1)

    sets = {
        "all": temporal,
        "shape": shape,
        "reduced32": temporal[reduced_cols],
        "spectral_acc16": spectral,
        "spectral_ilr_acc12": ilr_spectral,
        "shape_plus_spectral56": pd.concat([shape, spectral], axis=1),
    }
    return data, y, groups, sets


def _metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _record_confusion(confusion_rows, class_rows, feature_set, n_features, model_name, protocol, split, y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    for i, actual in enumerate(LABELS):
        total = int(matrix[i].sum())
        for j, predicted in enumerate(LABELS):
            count = int(matrix[i, j])
            confusion_rows.append({
                "feature_set": feature_set,
                "n_features": n_features,
                "protocol": protocol,
                "split": split,
                "model": model_name,
                "true_class": actual,
                "predicted_class": predicted,
                "count": count,
                "row_total": total,
                "row_pct": 100.0 * count / total if total else 0.0,
                "is_correct": actual == predicted,
            })

    report = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    for label in LABELS:
        class_rows.append({
            "feature_set": feature_set,
            "n_features": n_features,
            "protocol": protocol,
            "split": split,
            "model": model_name,
            "class": label,
            "precision": report[label]["precision"],
            "recall": report[label]["recall"],
            "f1": report[label]["f1-score"],
            "support": int(report[label]["support"]),
        })


def evaluate_set(feature_set, X, y, groups, confusion_rows, class_rows):
    summaries = []
    labels = LABELS

    for model_name, make_model in MODELS.items():
        model = make_model()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_true, cv_pred, cv_fold_metrics = [], [], []
        for fold_number, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            fitted = clone(model)
            fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
            prediction = fitted.predict(X.iloc[test_idx])
            actual = y.iloc[test_idx]
            cv_true.extend(actual.tolist())
            cv_pred.extend(prediction.tolist())
            fold_metrics = _metrics(actual, prediction)
            cv_fold_metrics.append(fold_metrics)
            summaries.append({
                "feature_set": feature_set,
                "n_features": X.shape[1],
                "model": model_name,
                "protocol": "stratified_5fold",
                "split": f"fold_{fold_number}",
                **fold_metrics,
            })

        cv_mean = pd.DataFrame(cv_fold_metrics).mean()
        cv_std = pd.DataFrame(cv_fold_metrics).std(ddof=0)
        summaries.append({
            "feature_set": feature_set,
            "n_features": X.shape[1],
            "model": model_name,
            "protocol": "stratified_5fold",
            "split": "mean",
            **{metric: float(cv_mean[metric]) for metric in cv_mean.index},
            **{f"{metric}_std": float(cv_std[metric]) for metric in cv_std.index},
        })
        cv_pooled = _metrics(cv_true, cv_pred)
        summaries.append({
            "feature_set": feature_set,
            "n_features": X.shape[1],
            "model": model_name,
            "protocol": "stratified_5fold",
            "split": "pooled",
            **cv_pooled,
        })
        _record_confusion(
            confusion_rows, class_rows, feature_set, X.shape[1], model_name,
            "stratified_5fold", "pooled", cv_true, cv_pred,
        )

        logo_true, logo_pred, logo_fold_metrics = [], [], []
        for rotation in sorted(groups.unique()):
            test_mask = (groups == rotation).to_numpy()
            fitted = clone(model)
            fitted.fit(X.loc[~test_mask], y.loc[~test_mask])
            prediction = fitted.predict(X.loc[test_mask])
            actual = y.loc[test_mask]
            fold_metrics = _metrics(actual, prediction)
            logo_fold_metrics.append(fold_metrics)
            logo_true.extend(actual.tolist())
            logo_pred.extend(prediction.tolist())
            split = f"{int(rotation)}Hz"
            summaries.append({
                "feature_set": feature_set,
                "n_features": X.shape[1],
                "model": model_name,
                "protocol": "leave_one_rotation_out",
                "split": split,
                **fold_metrics,
            })
            _record_confusion(
                confusion_rows, class_rows, feature_set, X.shape[1], model_name,
                "leave_one_rotation_out", split, actual, prediction,
            )

        logo_mean = pd.DataFrame(logo_fold_metrics).mean()
        logo_std = pd.DataFrame(logo_fold_metrics).std(ddof=0)
        summaries.append({
            "feature_set": feature_set,
            "n_features": X.shape[1],
            "model": model_name,
            "protocol": "leave_one_rotation_out",
            "split": "mean",
            **{metric: float(logo_mean[metric]) for metric in logo_mean.index},
            **{f"{metric}_std": float(logo_std[metric]) for metric in logo_std.index},
        })
        logo_pooled = _metrics(logo_true, logo_pred)
        summaries.append({
            "feature_set": feature_set,
            "n_features": X.shape[1],
            "model": model_name,
            "protocol": "leave_one_rotation_out",
            "split": "pooled",
            **logo_pooled,
        })
        _record_confusion(
            confusion_rows, class_rows, feature_set, X.shape[1], model_name,
            "leave_one_rotation_out", "pooled", logo_true, logo_pred,
        )

    return summaries


def evaluate_nested_feature_selection(y, groups, feature_sets, confusion_rows, class_rows):
    """Escolhe a representação por CV agrupada dentro de cada treino externo."""
    summaries = []
    selected_rows = []

    for model_name, make_model in MODELS.items():
        outer_true, outer_pred, outer_fold_metrics, outer_feature_counts = [], [], [], []
        for held_rotation in sorted(groups.unique()):
            outer_test_mask = (groups == held_rotation).to_numpy()
            outer_train_positions = np.flatnonzero(~outer_test_mask)
            y_train = y.iloc[outer_train_positions]
            group_train = groups.iloc[outer_train_positions]
            inner_cv = GroupKFold(n_splits=group_train.nunique())

            candidate_scores = {}
            for feature_set, X in feature_sets.items():
                X_train = X.iloc[outer_train_positions]
                fold_scores = []
                for inner_train, inner_valid in inner_cv.split(X_train, y_train, group_train):
                    fitted = clone(make_model())
                    fitted.fit(X_train.iloc[inner_train], y_train.iloc[inner_train])
                    prediction = fitted.predict(X_train.iloc[inner_valid])
                    fold_scores.append(
                        f1_score(
                            y_train.iloc[inner_valid], prediction,
                            average="macro", zero_division=0,
                        )
                    )
                candidate_scores[feature_set] = float(np.mean(fold_scores))

            chosen_set = max(candidate_scores, key=candidate_scores.get)
            chosen_X = feature_sets[chosen_set]
            fitted = clone(make_model())
            fitted.fit(chosen_X.iloc[outer_train_positions], y_train)
            prediction = fitted.predict(chosen_X.loc[outer_test_mask])
            actual = y.loc[outer_test_mask]
            metrics = _metrics(actual, prediction)
            split = f"{int(held_rotation)}Hz"
            n_features = chosen_X.shape[1]
            outer_true.extend(actual.tolist())
            outer_pred.extend(prediction.tolist())
            outer_fold_metrics.append(metrics)
            outer_feature_counts.append(n_features)
            selected_rows.append({
                "model": model_name,
                "held_out_rotation": split,
                "selected_feature_set": chosen_set,
                "n_features": n_features,
                "inner_mean_f1_macro": candidate_scores[chosen_set],
            })
            summaries.append({
                "feature_set": "nested_selector",
                "n_features": n_features,
                "model": model_name,
                "protocol": "leave_one_rotation_out_nested",
                "split": split,
                **metrics,
            })
            _record_confusion(
                confusion_rows, class_rows, "nested_selector", n_features, model_name,
                "leave_one_rotation_out_nested", split, actual, prediction,
            )

        fold_mean = pd.DataFrame(outer_fold_metrics).mean()
        fold_std = pd.DataFrame(outer_fold_metrics).std(ddof=0)
        summaries.append({
            "feature_set": "nested_selector",
            "n_features": float(np.mean(outer_feature_counts)),
            "model": model_name,
            "protocol": "leave_one_rotation_out_nested",
            "split": "mean",
            **{metric: float(fold_mean[metric]) for metric in fold_mean.index},
            **{f"{metric}_std": float(fold_std[metric]) for metric in fold_std.index},
        })
        pooled = _metrics(outer_true, outer_pred)
        summaries.append({
            "feature_set": "nested_selector",
            "n_features": float(np.mean(outer_feature_counts)),
            "model": model_name,
            "protocol": "leave_one_rotation_out_nested",
            "split": "pooled",
            **pooled,
        })
        _record_confusion(
            confusion_rows, class_rows, "nested_selector",
            int(round(np.mean(outer_feature_counts))), model_name,
            "leave_one_rotation_out_nested", "pooled", outer_true, outer_pred,
        )

    return summaries, selected_rows


def run_experiment(data_path: str | Path = SPECTRAL_FEATURES_CSV):
    _, y, groups, feature_sets = load_data(data_path)
    summaries, confusion_rows, class_rows = [], [], []
    with threadpool_limits(limits=1):
        for feature_set, X in feature_sets.items():
            print(f"Avaliando {feature_set}: {X.shape[1]} features")
            summaries.extend(
                evaluate_set(feature_set, X, y, groups, confusion_rows, class_rows)
            )
        nested_rows, selected_sets = evaluate_nested_feature_selection(
            y, groups, feature_sets, confusion_rows, class_rows
        )
        summaries.extend(nested_rows)

    summary = pd.DataFrame(summaries)
    confusion = pd.DataFrame(confusion_rows)
    classes = pd.DataFrame(class_rows)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = DOCS_DIR / "comparison_generalization.csv"
    confusion_path = DOCS_DIR / "confusion_matrices_generalization.csv"
    classes_path = DOCS_DIR / "class_metrics_generalization.csv"
    selected_path = DOCS_DIR / "selected_features_by_rotation.csv"
    summary.to_csv(summary_path, index=False)
    confusion.to_csv(confusion_path, index=False)
    classes.to_csv(classes_path, index=False)
    pd.DataFrame(selected_sets).to_csv(selected_path, index=False)
    print(f"Métricas: {summary_path}")
    print(f"Matrizes: {confusion_path}")
    print(f"Métricas por classe: {classes_path}")
    print(f"Features escolhidas somente no treino: {selected_path}")
    return summary, confusion, classes


if __name__ == "__main__":
    if not SPECTRAL_FEATURES_CSV.exists():
        raise SystemExit(
            "Tabela espectral não encontrada. Execute "
            "'python -m src.build_dataset --with-spectral' primeiro."
        )
    run_experiment()

