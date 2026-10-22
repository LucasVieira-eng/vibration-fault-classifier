# Vibration Fault Classifier

Projeto para explorar sinais de vibração, extrair características (*features*) e treinar um classificador de falhas.

## Estrutura

- `data/raw/`: dados brutos (`.lvm`); não são versionados no Git.
- `data/processed/`: features processadas em CSV ou Parquet; não são versionadas por padrão.
- `notebooks/`: exploração e experimentos.
- `src/`: código reutilizável para leitura, extração de features, treino e predição.
- `models/`: modelos treinados; não são versionados por padrão.
- `app/`: dashboard Streamlit.
- `docs/`: relatório e documentação.

## Começar

1. Crie e ative um ambiente virtual.
2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

3. Coloque os arquivos `.lvm` em `data/raw/`.
4. Explore os notebooks em `notebooks/`.
5. Implemente/execute o pipeline de features e treino conforme o formato dos seus dados.

## Comandos úteis

Executar o dashboard:

```bash
streamlit run app/dashboard.py
```

Os módulos em `src/` são pontos de partida e podem precisar ser adaptados ao formato real dos arquivos `.lvm` e aos rótulos de falha disponíveis.

## Dados e Git

Os diretórios de dados e modelos são mantidos no repositório por meio de arquivos `.gitkeep`, mas seus conteúdos são ignorados pelo Git. Não coloque dados privados ou licenciados no repositório.
