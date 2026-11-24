# Vibration Fault Classifier

Projeto para explorar sinais de vibração, extrair características (*features*) e treinar um classificador de falhas.

## Começar

1. Crie e ative um ambiente virtual.
2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

3. Coloque os arquivos `.lvm` em `data/raw/`. ([https://zenodo.org/records/14388093](https://zenodo.org/records/14388093))
4. Explore os notebooks em `notebooks/`.
5. Implemente/execute o pipeline de features e treino conforme o formato dos seus dados.

Os módulos em `src/` são pontos de partida e podem precisar ser adaptados ao formato real dos arquivos `.lvm` e aos rótulos de falha disponíveis.

## Estrutura

- `data/raw/`: dados brutos (`.lvm`); não são versionados no Git.
- `data/processed/`: features processadas em CSV ou Parquet; não são versionadas por padrão.
- `notebooks/`: exploração e experimentos.
- `src/`: código reutilizável para leitura, extração de features, treino e predição.
- `models/`: modelos treinados; não são versionados por padrão.
- `app/`: dashboard Streamlit.
- `docs/`: relatório e documentação.

```text
vibration-fault-classifier/
├── .gitignore
├── README.md
├── requirements.txt
├── app/
│   └── dashboard.py          (dashboard Streamlit para classificar um CSV de features)
├── data/
│   ├── raw/                  (.lvm baixados; fora do Git, só o .gitkeep)
│   └── processed/            (features.csv gerado; fora do Git, só o .gitkeep)
├── docs/
│   ├── eda.md                (relatório da análise exploratória)
│   ├── relatorio.md          (template do relatório final, ainda vazio)
│   └── figures/
│       └── eda/
│           ├── acceleration_rms_by_condition.png
│           ├── class_distribution.png
│           ├── fft_band_rms_acc_de_h.png
│           ├── fft_spectrum_20hz.png
│           ├── missing_values.png
│           ├── time_waveforms_20hz.png
│           └── velocity_rms_by_condition.png
├── models/                   (modelo .joblib treinado; fora do Git, só o .gitkeep)
├── notebooks/
│   ├── 01-exploracao.ipynb   (carrega os .lvm e inspeciona valores ausentes)
│   ├── 02-features.ipynb     (vazio)
│   └── 03-eda.ipynb          (EDA completa)
└── src/
    ├── __init__.py
    ├── download_data.py      (baixa e extrai os zips do Zenodo)
    ├── load.py               (lê um .lvm e adiciona type e rotation_hz)
    ├── features.py           (extrai as estatísticas de um sinal)
    ├── build_dataset.py      (monta e salva data/processed/features.csv)
    ├── train.py              (treina o Random Forest e salva o modelo)
    └── predict.py            (carrega o modelo e prediz)
```
