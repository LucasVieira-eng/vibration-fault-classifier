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
│   ├── 01-exploracao.ipynb     (carrega os .lvm e inspeciona valores ausentes)
│   └── 03-eda.ipynb            (EDA completa)
└── src/
    ├── config.py               (caminhos, colunas e constantes compartilhadas)
    ├── download_data.py        (baixa e extrai os zips do Zenodo)
    ├── load.py                 (lê um .lvm e adiciona as colunas type e rotation_hz)
    ├── features.py             (estatísticas de um sinal)
    ├── build_dataset.py        (monta e salva data/processed/features.csv)
    ├── evaluate.py             (carga de X/y, 5-fold, rotação deixada de fora, features de formato)
    ├── random_forest.py        (Random Forest)
    ├── hist_gradient_boosting.py   (Gradient Boosting por histograma)
    ├── logistic_regression.py  (Regressão Logística com padronização)
    ├── compare_models.py       (compara os três modelos nos dois protocolos)
    └── predict.py              (carrega um modelo salvo e prediz)
```

## `data/processed/features.csv`

Tabela de entrada dos modelos, gerada por `python -m src.build_dataset`. Ela tem **uma linha por arquivo `.lvm`** (1.200 linhas) e **67 colunas**: cada sinal bruto de 36.000 pontos por canal é resumido em algumas estatísticas.

### Colunas de identificação e rótulo

| Coluna | Tipo | Descrição |
|---|---|---|
| `file` | texto | Nome do arquivo `.lvm` de origem. Serve para rastrear a amostra; não é usada no treino. |
| `type` | texto | Condição da máquina: `Baseline`, `Bearing`, `Bearing+Unbalance`, `Misalignment_NDE`, `Unbalance_DE` ou `Unbalance_NDE`. É o **alvo** da classificação. Vem do nome da pasta filha de `data/raw/`. |
| `rotation_hz` | inteiro | Rotação da máquina em Hz (10, 20, 30 ou 40), tirada da pasta `<N>Hz`. Não é usada como feature por padrão; serve para agrupar o protocolo de "rotação deixada de fora". |

### Colunas de features (64)

Para cada um dos **8 canais** são calculadas **8 estatísticas**, nomeadas `<canal>_<estatística>` (por exemplo `acc_de_h_rms`, `vel_nde_v_kurtosis`).

**Canais:** `acc_de_h`, `acc_de_v`, `acc_nde_h`, `acc_nde_v` (aceleração) e `vel_de_h`, `vel_de_v`, `vel_nde_h`, `vel_nde_v` (velocidade). `de`/`nde` indicam o lado do eixo (*drive end* / *non-drive end*) e `h`/`v` a direção (horizontal / vertical).

**Estatísticas** (sobre os valores finitos do sinal, com `x` o vetor do canal):

| Sufixo | O que mede |
|---|---|
| `mean` | valor médio |
| `std` | desvio padrão |
| `rms` | raiz da média dos quadrados (intensidade do sinal) |
| `mean_abs` | média dos valores absolutos |
| `peak` | maior valor absoluto |
| `peak_to_peak` | máximo menos mínimo |
| `skewness` | assimetria da distribuição |
| `kurtosis` | curtose (excesso), sensível a impactos |

### Observações

- **Valores ausentes:** em cada arquivo, `time` e os canais de velocidade terminam com alguns `NaN` (de 1 a 310 linhas). O `load_lvm` descarta toda linha que tenha qualquer `NaN` antes de calcular as features, então cada arquivo é resumido com um pouco menos de 36.000 pontos.
- **Colunas usadas no treino:** `evaluate.load_xy` mantém só as colunas numéricas e retira `type` (alvo) e `rotation_hz`. Sobram as 64 features. `file` é texto e fica de fora.
- **Conjunto `shape` (40 colunas):** `evaluate.to_shape_features` deriva, a partir de `features.csv`, 5 features sem unidade por canal: `skewness`, `kurtosis`, fator de crista (`peak / rms`), fator de forma (`rms / mean_abs`) e fator de impulso (`peak / mean_abs`). Elas não crescem com a amplitude do sinal.
- **Regenerar:** se `load.py` ou `features.py` mudarem, rode `python -m src.build_dataset` de novo. O arquivo não é versionado no Git.

## Documentação

- [`docs/eda.md`](docs/eda.md): análise exploratória dos sinais.
- [`docs/relatorio.md`](docs/relatorio.md): metodologia, resultados, limitações e próximos passos.
