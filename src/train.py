"""Treino e avaliação de um classificador de falhas."""

from pathlib import Path
import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


def train(
    csv_path: str | Path,
    target_column: str,
    model_path: str | Path = "models/vibration_fault_model.joblib",
):
    data = pd.read_csv(csv_path)
    if target_column not in data.columns:
        raise KeyError(f"Coluna-alvo '{target_column}' não encontrada.")

    data = data.dropna(subset=[target_column])
    X = data.drop(columns=[target_column])
    X = X.select_dtypes(include="number")
    y = data[target_column]

    if X.empty:
        raise ValueError("Nenhuma coluna numérica de feature foi encontrada.")
    if y.nunique() < 2:
        raise ValueError("O treino precisa de pelo menos duas classes.")

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    model = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Acurácia: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions, zero_division=0))

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": list(X.columns)}, model_path)
    print(f"Modelo salvo em: {model_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV com features e rótulo")
    parser.add_argument("--target", required=True, help="Nome da coluna-alvo")
    parser.add_argument(
        "--model", default="models/vibration_fault_model.joblib"
    )
    args = parser.parse_args()
    train(args.csv, args.target, args.model)
