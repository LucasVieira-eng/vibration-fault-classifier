"""Predição de classes de falha usando um modelo salvo."""

from pathlib import Path
import joblib
import pandas as pd


def predict(
    features: pd.DataFrame,
    model_path: str | Path = "models/vibration_fault_model.joblib",
):
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    missing = set(feature_columns) - set(features.columns)
    if missing:
        raise ValueError(f"Features ausentes: {sorted(missing)}")

    return model.predict(features[feature_columns])


if __name__ == "__main__":
    raise SystemExit(
        "Importe predict() em um script ou notebook e passe um DataFrame de features."
    )
