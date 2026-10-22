"""Dashboard inicial para testar predições a partir de um CSV de features."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Permite importar o pacote src quando o app é executado a partir da raiz.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import predict  # noqa: E402

st.set_page_config(page_title="Vibration Fault Classifier", layout="wide")
st.title("Vibration Fault Classifier")
st.write("Envie um CSV com as features numéricas esperadas pelo modelo treinado.")

model_path = st.text_input(
    "Caminho do modelo", "models/vibration_fault_model.joblib"
)
uploaded = st.file_uploader("CSV de features", type=["csv"])

if uploaded is not None:
    features = pd.read_csv(uploaded)
    st.subheader("Dados enviados")
    st.dataframe(features.head())

    if st.button("Classificar"):
        try:
            predictions = predict(features, model_path)
            result = features.copy()
            result["predicted_fault"] = predictions
            st.subheader("Predições")
            st.dataframe(result)
            st.download_button(
                "Baixar resultados em CSV",
                result.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Não foi possível classificar os dados: {exc}")
else:
    st.info("Carregue um CSV de features para começar.")
