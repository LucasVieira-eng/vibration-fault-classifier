# Relatório — Vibration Fault Classifier

## Objetivo

Comparar três modelos de classificação (Random Forest, Gradient Boosting e Regressão Logística) na tarefa de identificar a condição de uma máquina rotativa — sem falha ou com um de cinco tipos de falha — a partir de sinais de vibração. O objetivo é **comparar o comportamento dos modelos com os mesmos dados e os mesmos protocolos de avaliação**, e não escolher um único melhor modelo; por isso nenhum modelo foi descartado e os hiperparâmetros não foram ajustados.

## Conjunto de dados

- **Origem:** [Vibration signals of different fault conditions in a generic rotating machine](https://zenodo.org/records/14388093), registro 14388093 do Zenodo.
- **Número de amostras:** 1.200 arquivos `.lvm`, cada um com 36.000 linhas e 11 colunas.
- **Frequência de amostragem:** 6 kHz (cerca de 6 s por arquivo).
- **Canais/sensores:** quatro canais de aceleração (`acc_de_h`, `acc_de_v`, `acc_nde_h`, `acc_nde_v`) e quatro de velocidade (`vel_de_h`, `vel_de_v`, `vel_nde_h`, `vel_nde_v`), além de ponto, tempo e *trigger*. `de`/`nde` indicam o lado do eixo (*drive end* / *non-drive end*) e `h`/`v` a direção (horizontal / vertical).
- **Classes e distribuição:** seis condições, com 200 arquivos cada (`Baseline`, `Bearing`, `Bearing+Unbalance`, `Misalignment_NDE`, `Unbalance_DE`, `Unbalance_NDE`). Cada condição tem 50 arquivos em cada uma das quatro rotações (10, 20, 30 e 40 Hz). As classes são perfeitamente balanceadas.
- **Valores ausentes:** `time` e os canais de velocidade têm `NaN` no final de cada arquivo (de 1 a 310 linhas, em média cerca de 111). A análise completa está em [`eda.md`](eda.md).

## Metodologia

1. **Leitura e validação (`src/load.py`).** Cada `.lvm` é lido pulando os dois cabeçalhos do LabVIEW. Linhas com qualquer valor ausente são removidas. O rótulo (`type`) é o nome da pasta filha de `data/raw/` e a rotação (`rotation_hz`) vem da pasta `<N>Hz` do arquivo.
2. **Exploração dos sinais.** Resumida em [`eda.md`](eda.md). Um achado relevante para o que segue: a amplitude da vibração cresce muito com a rotação (RMS mediano em `acc_de_h` de 0,42 a 10 Hz para 2,35 a 40 Hz).
3. **Extração de características (`src/features.py`, `src/build_dataset.py`).** Para cada canal, oito estatísticas no domínio do tempo (média, desvio padrão, RMS, média absoluta, pico, pico a pico, assimetria e curtose), o que dá 64 features por arquivo. A tabela final (`features.csv`) tem uma linha por arquivo; os campos estão descritos no [README](../README.md). Não foram usadas features no domínio da frequência.
   - **Variante `shape` (40 features):** mantém só medidas de formato, sem unidade e que não crescem com a amplitude: assimetria, curtose, fator de crista (`pico / RMS`), fator de forma (`RMS / média absoluta`) e fator de impulso (`pico / média absoluta`).
4. **Modelos (`src/random_forest.py`, `src/hist_gradient_boosting.py`, `src/logistic_regression.py`).** Todos com hiperparâmetros padrão e `random_state=42`:
   - Random Forest com 200 árvores;
   - `HistGradientBoostingClassifier`;
   - Regressão Logística com `StandardScaler` em um `Pipeline` (o padronizador é ajustado só nos dados de treino de cada divisão) e `max_iter=2000`.
5. **Avaliação (`src/evaluate.py`, `src/compare_models.py`).** Dois protocolos, iguais para os três modelos. `rotation_hz` **não** é usada como feature, exceto no experimento indicado abaixo.