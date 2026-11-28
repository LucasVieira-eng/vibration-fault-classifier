
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SETS = {
    "all": "Estatísticas dos sinais (64 features)",
    "all_with_hz": "Estatísticas + rotação (65 features)",
    "shape": "Formato do sinal (40 features)",
    "reduced32": "Temporais reduzidas (32 features)",
    "spectral_acc16": "Espectrais absolutas — aceleração (16 features)",
    "spectral_ilr_acc12": "Composição espectral ILR — aceleração (12 features)",
    "shape_plus_spectral56": "Formato + espectrais (56 features)",
    "nested_selector": "Seleção de features dentro do treino",
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
        for key in ("all", "all_with_hz", "shape")
        if (DOCS / f"comparison_{key}.csv").exists()
    }

    generalization_path = DOCS / "comparison_generalization.csv"
    if generalization_path.exists():
        generalization = pd.read_csv(generalization_path)
        for feature_set, table in generalization.groupby("feature_set"):
            rows = []
            for model in table["model"].unique():
                model_rows = table[table["model"] == model]
                cv = model_rows[
                    (model_rows["protocol"] == "stratified_5fold")
                    & (model_rows["split"] == "mean")
                ]
                logo = model_rows[
                    (model_rows["protocol"].isin([
                        "leave_one_rotation_out", "leave_one_rotation_out_nested"
                    ]))
                    & (model_rows["split"] == "mean")
                ]
                if logo.empty:
                    continue
                logo = logo.iloc[0]
                cv = cv.iloc[0] if not cv.empty else None
                row = {
                    "model": model,
                    "cv_accuracy": cv["accuracy"] if cv is not None else np.nan,
                    "cv_f1_macro": cv["f1_macro"] if cv is not None else np.nan,
                    "cv_precision_macro": cv["precision_macro"] if cv is not None else np.nan,
                    "cv_recall_macro": cv["recall_macro"] if cv is not None else np.nan,
                    "cv_accuracy_std": cv.get("accuracy_std") if cv is not None else np.nan,
                    "cv_f1_macro_std": cv.get("f1_macro_std") if cv is not None else np.nan,
                    "leave_out_accuracy": logo["accuracy"],
                    "leave_out_f1_macro": logo["f1_macro"],
                    "leave_out_precision_macro": logo["precision_macro"],
                    "leave_out_recall_macro": logo["recall_macro"],
                    "leave_out_accuracy_std": logo.get("accuracy_std"),
                    "leave_out_f1_macro_std": logo.get("f1_macro_std"),
                }
                logo_protocol = (
                    "leave_one_rotation_out_nested"
                    if feature_set == "nested_selector"
                    else "leave_one_rotation_out"
                )
                for rotation in (10, 20, 30, 40):
                    speed = model_rows[
                        (model_rows["protocol"] == logo_protocol)
                        & (model_rows["split"] == f"{rotation}Hz")
                    ]
                    if not speed.empty:
                        row[f"leave_out_{rotation}Hz_acc"] = speed.iloc[0]["accuracy"]
                rows.append(row)
            if rows:
                results[feature_set] = pd.DataFrame(rows)

    matrices_path = DOCS / "confusion_matrices_generalization.csv"
    confusion = pd.read_csv(matrices_path) if matrices_path.exists() else pd.DataFrame()
    legacy_path = DOCS / "confusion_matrices.csv"
    if legacy_path.exists():
        legacy = pd.read_csv(legacy_path)
        legacy = legacy[legacy["feature_set"] == "all_with_hz"].copy()
        if not legacy.empty:
            legacy["split"] = "pooled"
            confusion = pd.concat([confusion, legacy], ignore_index=True)

    classes_path = DOCS / "class_metrics_generalization.csv"
    classes = pd.read_csv(classes_path) if classes_path.exists() else pd.DataFrame()
    return results, confusion, classes


def metrics_table(results, chosen_sets):
    rows = []
    for key in chosen_sets:
        for _, row in results[key].iterrows():
            common = {
                "Conjunto": SETS[key],
                "Modelo": MODEL_NAMES.get(row["model"], row["model"]),
            }
            if pd.notna(row.get("cv_accuracy")):
                rows.append({**common, "Protocolo": "Validação cruzada 5-fold",
                             "Acurácia": row["cv_accuracy"], "F1 macro": row["cv_f1_macro"]})
            logo_label = (
                "Seleção interna + rotação deixada de fora"
                if key == "nested_selector"
                else "Rotação deixada de fora"
            )
            rows.append({**common, "Protocolo": logo_label,
                         "Acurácia": row["leave_out_accuracy"],
                         "F1 macro": row["leave_out_f1_macro"]})
    return pd.DataFrame(rows)


st.set_page_config(page_title="Resultados | Classificador de Falhas", layout="wide")
st.title("Resultados da classificação de falhas")
st.caption("Comparação dos modelos, conjuntos de características e protocolos com os CSVs em docs/.")

results, confusion, class_metrics = load_data()
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
logo = metrics[metrics["Protocolo"].isin([
    "Rotação deixada de fora", "Seleção interna + rotação deixada de fora"
])]
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

overview, by_speed, matrices, by_class, notes = st.tabs(
    ["Visão geral", "Por rotação", "Matrizes de confusão", "Por classe", "Dados e interpretação"]
)

with overview:
    metric = st.radio("Métrica", ["Acurácia", "F1 macro"], horizontal=True)
    chart = metrics.copy()
    chart["Série"] = chart["Conjunto"] + " · " + chart["Modelo"]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=chart, x="Série", y=metric, hue="Protocolo", errorbar=None, ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=65)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    with st.expander("Tabela completa de métricas"):
        st.dataframe(metrics.style.format({"Acurácia": "{:.2%}", "F1 macro": "{:.2%}"}),
                     width="stretch", hide_index=True)

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
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.dataframe(speed[speed["Conjunto"] == selected].pivot(
            index="Modelo", columns="Rotação", values="Acurácia"
        ).style.format("{:.1%}"), width="stretch")

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
            protocols = confusion.loc[confusion["feature_set"] == key, "protocol"].unique()
            protocol = st.selectbox(
                "Protocolo",
                protocols,
                format_func=lambda value: {
                    "stratified_5fold": "5-fold",
                    "leave_one_rotation_out": "Rotação deixada de fora",
                    "leave_one_rotation_out_nested": "Seleção interna + rotação deixada de fora",
                }.get(value, value),
            )
            selected_protocol = confusion[
                (confusion["feature_set"] == key) & (confusion["protocol"] == protocol)
            ]
            splits = list(dict.fromkeys(selected_protocol["split"].tolist()))
            split = st.selectbox(
                "Divisão", splits,
                index=splits.index("pooled") if "pooled" in splits else 0,
            )
            models_for_selection = [m for m in MODELS if m in selected_protocol["model"].unique()]
            model = st.selectbox("Modelo", models_for_selection)
            selected = selected_protocol[
                (selected_protocol["model"] == model) & (selected_protocol["split"] == split)
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
            st.pyplot(fig, width="stretch")
            plt.close(fig)

with by_class:
    st.subheader("Precisão, revocação e F1 por condição")
    if class_metrics.empty:
        st.info("As métricas por classe aparecem depois de gerar os novos resultados.")
    else:
        class_sets = [key for key in chosen if key in class_metrics["feature_set"].unique()]
        if not class_sets:
            st.info("As métricas por classe ainda não estão disponíveis para essa seleção.")
        else:
            key = st.selectbox("Conjunto de características", class_sets, format_func=lambda x: SETS[x], key="class_set")
            model = st.selectbox("Modelo", [m for m in MODELS if m in class_metrics["model"].unique()], key="class_model")
            available_protocols = class_metrics.loc[class_metrics["feature_set"] == key, "protocol"].unique()
            protocol = st.selectbox(
                "Protocolo",
                available_protocols,
                format_func=lambda value: {
                    "stratified_5fold": "5-fold",
                    "leave_one_rotation_out": "Rotação deixada de fora",
                    "leave_one_rotation_out_nested": "Seleção interna + rotação deixada de fora",
                }.get(value, value),
                key="class_protocol",
            )
            class_rows = class_metrics[
                (class_metrics["feature_set"] == key)
                & (class_metrics["model"] == model)
                & (class_metrics["protocol"] == protocol)
            ]
            split_options = list(dict.fromkeys(class_rows["split"].tolist()))
            split = st.selectbox("Divisão", split_options, index=split_options.index("pooled") if "pooled" in split_options else 0, key="class_split")
            class_rows = class_rows[class_rows["split"] == split]
            class_metric = st.radio("Métrica por classe", ["precision", "recall", "f1"], horizontal=True)
            fig, ax = plt.subplots(figsize=(10, 4.5))
            sns.barplot(data=class_rows, x="class", y=class_metric, color="#4472C4", ax=ax)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("")
            ax.set_ylabel(class_metric.replace("precision", "Precisão").replace("recall", "Revocação").replace("f1", "F1"))
            ax.tick_params(axis="x", rotation=35)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            st.dataframe(class_rows, width="stretch", hide_index=True)

with notes:
    st.markdown(
        """
        - **Estatísticas dos sinais (64):** estatísticas calculadas nos oito canais de vibração.
        - **Estatísticas + rotação (65):** as 64 estatísticas mais a coluna rotation_hz.
          Isso não significa que a forma de onda bruta foi adicionada como feature.
        - **Formato do sinal (40):** cinco características derivadas por canal, incluindo
          curtose e fatores de crista, forma e impulso.
        - **Temporais reduzidas (32):** RMS, pico a pico, assimetria e curtose por canal.
        - **Espectrais absolutas (16):** energia em quatro bandas nos quatro acelerômetros.
        - **Espectrais ILR (12):** três razões logarítmicas independentes entre as quatro
          bandas de cada acelerômetro, reduzindo a redundância da composição.
        - **5-fold:** cada divisão mistura amostras das rotações disponíveis.
        - **Rotação deixada de fora:** cada velocidade é testada após treinar nas outras três.
        - **Seleção interna:** para cada teste de rotação, o painel mostra o resultado de
          um processo que escolhe as features por validação agrupada nas outras velocidades.
        - O painel lê os CSVs; para gerar resultados atualizados, execute o pipeline documentado.
        - Na tabela, F1 macro da rotação é a média entre velocidades; a matriz pooled agrega
          as previsões das quatro velocidades, então esses valores não precisam coincidir.
        - As quatro rotações já foram examinadas durante o desenvolvimento; os resultados
          ainda precisam de confirmação em uma aquisição independente.
        """
    )
    st.write("Métricas: comparison_all.csv, comparison_all_with_hz.csv e comparison_shape.csv.")
    st.write("Experimentos novos: comparison_generalization.csv e arquivos de suporte generalization/class_metrics.")
