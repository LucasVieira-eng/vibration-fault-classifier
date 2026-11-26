# Vibration Fault Classifier

Projeto para classificar falhas em uma máquina rotativa a partir de sinais de vibração. Os sinais brutos (`.lvm`) são resumidos em estatísticas por arquivo (*features*), e três modelos de classificação são treinados e comparados com os mesmos protocolos de avaliação.

- **Dados:** [Vibration signals of different fault conditions in a generic rotating machine](https://zenodo.org/records/14388093) (Zenodo): 1.200 sinais, 6 condições (Baseline e 5 tipos de falha) e 4 rotações (10, 20, 30 e 40 Hz).
- **Modelos:** Random Forest, Gradient Boosting (`HistGradientBoostingClassifier`) e Regressão Logística, todos com hiperparâmetros padrão. O objetivo é comparar, não escolher um vencedor.
- **Avaliação:** validação cruzada estratificada de 5 partes e "rotação deixada de fora" (treina em três rotações e testa na quarta).
- **Resultado em uma frase:** os três modelos chegam a ~99% na validação cruzada, mas caem para 26–44% quando testados em uma rotação que não viram no treino. Detalhes em [`docs/relatorio.md`](docs/relatorio.md).

## Começar

1. Crie e ative um ambiente virtual.
2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

3. Baixe os dados brutos para `data/raw/` (um `.zip` por condição, extraído em `data/raw/<condição>/<N>Hz/`):

   ```bash
   python -m src.download_data
   ```

4. Gere a tabela de features (`data/processed/features.csv`):

   ```bash
   python -m src.build_dataset
   ```

5. Compare os três modelos nos dois protocolos de avaliação:

   ```bash
   python -m src.compare_models --features all     # 64 features (todas as estatísticas)
   python -m src.compare_models --features shape   # 40 features (só formato do sinal)
   ```

   Cada execução imprime a tabela de métricas e as matrizes de confusão, e salva a tabela em `docs/comparison_<conjunto>.csv`.

Todos os comandos devem ser executados a partir da raiz do projeto, como módulo (`python -m src.<nome>`).

Cada modelo também pode ser executado sozinho (`python -m src.random_forest`, `python -m src.hist_gradient_boosting`, `python -m src.logistic_regression`), o que imprime as métricas da validação cruzada de 5 partes. Para salvar um modelo treinado em `models/`, chame `build_model(save_model=True)` a partir de Python; o dashboard usa esse arquivo.

## Estrutura

```text
vibration-fault-classifier/
├── README.md
├── requirements.txt
├── app/
│   └── dashboard.py            (dashboard Streamlit: classifica um CSV de features com um modelo salvo)
├── data/
│   ├── raw/                    (.lvm baixados; fora do Git)
│   └── processed/              (features.csv gerado; fora do Git)
├── docs/
│   ├── eda.md                  (análise exploratória dos sinais)
│   ├── relatorio.md            (metodologia, resultados e limitações)
│   ├── comparison_all.csv      (métricas: 64 features, sem rotation_hz)
│   ├── comparison_shape.csv    (métricas: 40 features de formato)
│   ├── comparison_all_with_hz.csv  (métricas: 64 features + rotation_hz)
│   └── figures/eda/            (figuras da EDA)
├── models/                     (modelos .joblib treinados; fora do Git)
├── notebooks/