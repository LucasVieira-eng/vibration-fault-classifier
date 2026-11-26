import pandas as pd
from src import hist_gradient_boosting, logistic_regression, random_forest
from src.config import PROJECT_ROOT
from src.evaluate import cross_validate_model, leave_one_group_out, load_xy

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

MODELS = {
    "Random Forest": random_forest.make_model,
    "Hist Gradient Boosting": hist_gradient_boosting.make_model,
    "Logistic Regression": logistic_regression.make_model,
}
OUTPUT_PATH = PROJECT_ROOT / "docs" / "comparison.csv"


def compare_models():
    X, y, groups = load_xy(return_groups=True)
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
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_PATH)

    print(table.round(4).T)
    for name, matrix in confusions.items():
        print(f"\nMatriz de confusão (rotação deixada de fora) - {name}")
        print(matrix)
    return table


if __name__ == "__main__":
    compare_models()