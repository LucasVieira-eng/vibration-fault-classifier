from pathlib import Path
import argparse
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from src.config import FEATURES_CSV, MODELS_DIR, TARGET_COLUMN
from src.evaluate import cross_validate_model, load_xy

pd.set_option("display.max_columns", None)

MODEL_PATH = MODELS_DIR / "hist_gradient_boosting.joblib"


def build_model(
    csv_path: str | Path = FEATURES_CSV,
    target_column: str = TARGET_COLUMN,
    model_path: str | Path = MODEL_PATH,
    show_results: bool = True,
    save_model: bool = False,
):
    X, y = load_xy(csv_path, target_column)

    model = HistGradientBoostingClassifier(random_state=42)
    results = cross_validate_model(model, X, y)

    model.fit(X, y)

    if show_results:
        print(results)
    if save_model:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "feature_columns": list(X.columns)}, model_path)
        print(f"Modelo salvo em: {model_path}")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=FEATURES_CSV, help="CSV com features e rótulo")
    parser.add_argument("--target", default=TARGET_COLUMN, help="Nome da coluna-alvo")
    parser.add_argument("--model", default=MODEL_PATH)
    args = parser.parse_args()
    build_model(args.csv, args.target, args.model, save_model=False)