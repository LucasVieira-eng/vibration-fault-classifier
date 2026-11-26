
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SETS = {
    "all": "Estatísticas dos sinais (64 features)",
    "all_with_hz": "Estatísticas + rotação (65 features)",
    "shape": "Formato do sinal (40 features)",
}
MODELS = ["Random Forest", "Hist Gradient Boosting", "Logistic Regression"]
MODEL_NAMES = {
    "Random Forest": "Random Forest",
    "Hist Gradient Boosting": "Gradient Boosting por histograma",
    "Logistic Regression": "Regressão Logística",
}


@st.cache_data
def load_data():
    results = {
        key: pd.read_csv(DOCS / f"comparison_{key}.csv")
        for key in SETS
        if (DOCS / f"comparison_{key}.csv").exists()
    }
    path = DOCS / "confusion_matrices.csv"
    confusion = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return results, confusion


def metrics_table(results, chosen_sets):
    rows = []
    for key in chosen_sets:
        for _, row in results[key].iterrows():
            common = {
                "Conjunto": SETS[key],
                "Modelo": MODEL_NAMES.get(row["model"], row["model"]),
            }
            rows.append({**common, "Protocolo": "Validação cruzada 5-fold",
                         "Acurácia": row["cv_accuracy"], "F1 macro": row["cv_f1_macro"]})
            rows.append({**common, "Protocolo": "Rotação deixada de fora",
                         "Acurácia": row["leave_out_accuracy"],
                         "F1 macro": row["leave_out_f1_macro"]})
    return pd.DataFrame(rows)


st.set_page_config(page_title="Resultados | Classificador de Falhas", layout="wide")
st.title("Resultados da classificação de falhas")
st.caption("Comparação dos modelos, conjuntos de características e protocolos com os CSVs em docs/.")

results, confusion = load_data()
available = [key for key in SETS if key in results]
if not available:
    st.error("Não encontrei os arquivos comparison_*.csv na pasta docs/.")
    st.stop()

chosen = st.multiselect("Conjuntos de características", available, default=available,
                        format_func=lambda key: SETS[key])
metrics = metrics_table(results, chosen)
if metrics.empty:
    st.info("Selecione pelo menos um conjunto de características.")
    st.stop()

cv = metrics[metrics["Protocolo"] == "Validação cruzada 5-fold"]
logo = metrics[metrics["Protocolo"] == "Rotação deixada de fora"]
best_cv = cv.loc[cv["Acurácia"].idxmax()]
best_logo = logo.loc[logo["Acurácia"].idxmax()]
a, b, c = st.columns(3)
a.metric("Melhor acurácia — 5-fold", f"{best_cv['Acurácia']:.1%}",
         f"{best_cv['Modelo']} · {best_cv['Conjunto']}")
b.metric("Melhor acurácia — rotação inédita", f"{best_logo['Acurácia']:.1%}",
         f"{best_logo['Modelo']} · {best_logo['Conjunto']}")
c.metric("Melhor F1 macro — rotação inédita", f"{logo['F1 macro'].max():.1%}",
         "média reportada nos CSVs")

st.warning(
    "Nos dois testes, o modelo recebe arquivos diferentes para treinar e avaliar. A diferença é que, no 5-fold, ele já viu no treinamento todas as velocidades que aparecem na avaliação. No teste de rotação deixada de fora, reservamos uma velocidade inteira: o modelo treina sem nenhum arquivo daquela velocidade e só a vê na hora da avaliação."
)

overview, by_speed, matrices, notes = st.tabs(
    ["Visão geral", "Por rotação", "Matrizes de confusão", "Dados e interpretação"]
)

with overview:
    metric = st.radio("Métrica", ["Acurácia", "F1 macro"], horizontal=True)
    chart = metrics.copy()
    chart["Série"] = chart["Conjunto"] + " · " + chart["Modelo"]
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=chart, x="Série", y=metric, hue="Protocolo", errorbar=None, ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=65)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    with st.expander("Tabela completa de métricas"):
        st.dataframe(metrics.style.format({"Acurácia": "{:.2%}", "F1 macro": "{:.2%}"}),
                     use_container_width=True, hide_index=True)

with by_speed:
    st.subheader("Acurácia em cada rotação deixada de fora")
    rows = []
    for key in chosen:
        for _, result in results[key].iterrows():
            for hz in (10, 20, 30, 40):
                col = f"leave_out_{hz}Hz_acc"
                if col in result:
                    rows.append({"Conjunto": SETS[key],
                                 "Modelo": MODEL_NAMES.get(result["model"], result["model"]),
                                 "Rotação": f"{hz} Hz", "Acurácia": result[col]})
    speed = pd.DataFrame(rows)
    if speed.empty:
        st.info("Não há métricas separadas por rotação nos arquivos selecionados.")
    else:
        selected = st.selectbox("Conjunto", sorted(speed["Conjunto"].unique()))
        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.barplot(data=speed[speed["Conjunto"] == selected], x="Rotação",
                    y="Acurácia", hue="Modelo", errorbar=None, ax=ax)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.dataframe(speed[speed["Conjunto"] == selected].pivot(
            index="Modelo", columns="Rotação", values="Acurácia"
        ).style.format("{:.1%}"), use_container_width=True)

with matrices:
    st.subheader("Erros no protocolo de rotação deixada de fora")
    st.caption("As matrizes agregam as previsões das quatro rotações testadas; não são matrizes do 5-fold.")
    if confusion.empty:
        st.info("Não encontrei docs/confusion_matrices.csv.")
    else:
        matrix_sets = [key for key in chosen if key in confusion["feature_set"].unique()]
        if not matrix_sets:
            st.info("Não há matrizes para esta seleção.")
        else:
            key = st.selectbox("Conjunto para matriz", matrix_sets, format_func=lambda x: SETS[x])
            model = st.selectbox("Modelo", [m for m in MODELS if m in confusion["model"].unique()])
            selected = confusion[
                (confusion["feature_set"] == key)
                & (confusion["model"] == model)
                & (confusion["protocol"] == "leave_one_rotation_out")
            ]
            matrix = selected.pivot(index="true_class", columns="predicted_class", values="row_pct")
            matrix = matrix.reindex(index=sorted(matrix.index), columns=sorted(matrix.columns))
            fig, ax = plt.subplots(figsize=(9, 7))
            sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=100,
                        square=True, cbar_kws={"label": "% das amostras reais da classe"}, ax=ax)
            ax.set_xlabel("Classe prevista")
            ax.set_ylabel("Classe real")
            ax.set_title(MODEL_NAMES.get(model, model))
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

with notes:
    st.markdown(
        """
        - **Estatísticas dos sinais (64):** estatísticas calculadas nos oito canais de vibração.
        - **Estatísticas + rotação (65):** as 64 estatísticas mais a coluna rotation_hz.
          Isso não significa que a forma de onda bruta foi adicionada como feature.
        - **Formato do sinal (40):** cinco características derivadas por canal, incluindo
          curtose e fatores de crista, forma e impulso.
        - **5-fold:** cada divisão mistura amostras das rotações disponíveis.
        - **Rotação deixada de fora:** cada velocidade é testada após treinar nas outras três.
        - As visualizações leem os CSVs existentes; não treinam modelos nem recalculam métricas.
        """
    )
    st.write("Métricas: comparison_all.csv, comparison_all_with_hz.csv e comparison_shape.csv.")
    st.write("Matrizes: confusion_matrices.csv; somente rotação deixada de fora.")

