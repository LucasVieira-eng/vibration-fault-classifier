"""Compara os três modelos nos dois protocolos: 5-fold e rotação deixada de fora."""

import argparse

import pandas as pd
from src import hist_gradient_boosting, logistic_regression, random_forest
from src.config import PROJECT_ROOT
from src.evaluate import (
    cross_validate_model,
    leave_one_group_out,
    load_xy,
    to_shape_features,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

MODELS = {
    "Random Forest": random_forest.make_model,
    "Hist Gradient Boosting": hist_gradient_boosting.make_model,
    "Logistic Regression": logistic_regression.make_model,
}
DOCS_DIR = PROJECT_ROOT / "docs"


def compare_models(feature_set: str = "all"):
    """feature_set: 'all' (todas as estatísticas) ou 'shape' (só features de forma)."""
    X, y, groups = load_xy(return_groups=True)
    if feature_set == "shape":
        X = to_shape_features(X)
    print(f"Conjunto de features: {feature_set} ({X.shape[1]} colunas)")

    rows, confusions = [], {}
    for name, make_model in MODELS.items():
        cv = cross_validate_model(make_model(), X, y)
        logo = leave_one_group_out(make_model(), X, y, groups)

        row = {
            "model": name,
            "cv_accuracy": cv["accuracy_mean"],
            "cv_f1_macro": cv["f1_macro_mean"],
        }
        for hz, acc in logo["per_group"]["accuracy"].items():
            row[f"leave_out_{int(hz)}Hz_acc"] = acc
        row["leave_out_accuracy"] = logo["accuracy_mean"]
        row["leave_out_f1_macro"] = logo["f1_macro_mean"]
        rows.append(row)
        confusions[name] = logo["confusion_matrix"]

    table = pd.DataFrame(rows).set_index("model")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(DOCS_DIR / f"comparison_{feature_set}.csv")

    print(table.round(4).T)
    for name, matrix in confusions.items():
        print(f"\nMatriz de confusão (rotação deixada de fora) - {name}")
        print(matrix)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=["all", "shape"], default="all")
    args = parser.parse_args()
    compare_models(args.features)