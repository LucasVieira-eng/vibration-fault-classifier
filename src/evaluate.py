from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from src.config import FEATURES_CSV, TARGET_COLUMN


def load_xy(
    csv_path: str | Path = FEATURES_CSV,
    target_column: str = TARGET_COLUMN,
    drop_columns: tuple[str, ...] = ("rotation_hz",),
):
    """Lê o CSV de features e devolve X (só numéricas) e y."""
    data = pd.read_csv(csv_path)
    if target_column not in data.columns:
        raise KeyError(f"Coluna-alvo '{target_column}' não encontrada.")

    data = data.dropna(subset=[target_column])
    X = data.drop(columns=[target_column, *drop_columns])
    X = X.select_dtypes(include="number")
    y = data[target_column]

    if X.empty:
        raise ValueError("Nenhuma coluna numérica de feature foi encontrada.")
    if y.nunique() < 2:
        raise ValueError("O treino precisa de pelo menos duas classes.")

    return X, y


def cross_validate_model(model, X, y, n_splits: int = 5) -> dict:
    """Validação cruzada estratificada; devolve métricas e matriz de confusão."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies, f1s = [], []
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in cv.split(X, y):
        fold_model = clone(model)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = fold_model.predict(X.iloc[test_idx])
        y_true = y.iloc[test_idx]

        accuracies.append(accuracy_score(y_true, y_pred))
        f1s.append(f1_score(y_true, y_pred, average="macro"))
        y_true_all.append(y_true.to_numpy())
        y_pred_all.append(y_pred)

    labels = sorted(y.unique())
    matrix = confusion_matrix(
        np.concatenate(y_true_all), np.concatenate(y_pred_all), labels=labels
    )

    return {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "f1_macro_mean": float(np.mean(f1s)),
        "f1_macro_std": float(np.std(f1s)),
        "confusion_matrix": pd.DataFrame(matrix, index=labels, columns=labels),
    }