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
   - **Validação cruzada estratificada de 5 partes** (com embaralhamento): mede o desempenho em arquivos de rotações que o modelo já viu no treino.
   - **Rotação deixada de fora:** treina em três rotações e testa na quarta (300 arquivos, 50 por classe), repetindo para cada uma das quatro rotações. Mede a capacidade de generalizar para uma velocidade de operação nova. A acurácia reportada é a média das quatro rotações.
   - **Métricas:** acurácia e F1 macro (que trata as seis classes igualmente). Matrizes de confusão acumuladas das quatro rotações deixadas de fora (cada linha soma 200). O acaso para seis classes é 16,7% de acurácia.

## Resultados

### Todas as estatísticas (64 features), sem `rotation_hz`

Fonte: [`comparison_all.csv`](comparison_all.csv).

| Modelo | 5-fold: acurácia | 5-fold: F1 macro | Fora 10 Hz | Fora 20 Hz | Fora 30 Hz | Fora 40 Hz | Rotação fora: acurácia média | Rotação fora: F1 macro |
|---|---|---|---|---|---|---|---|---|
| Random Forest | 0,9983 | 0,9983 | 0,320 | 0,350 | 0,500 | 0,180 | 0,3375 | 0,2446 |
| Hist Gradient Boosting | 0,9933 | 0,9933 | 0,330 | 0,467 | 0,410 | 0,253 | 0,3650 | 0,2582 |
| Regressão Logística | 0,9917 | 0,9917 | 0,357 | 0,173 | 0,400 | 0,357 | 0,3217 | 0,2691 |

### Apenas features de formato (40 features), sem `rotation_hz`

Fonte: [`comparison_shape.csv`](comparison_shape.csv).

| Modelo | 5-fold: acurácia | 5-fold: F1 macro | Fora 10 Hz | Fora 20 Hz | Fora 30 Hz | Fora 40 Hz | Rotação fora: acurácia média | Rotação fora: F1 macro |
|---|---|---|---|---|---|---|---|---|
| Random Forest | 0,9967 | 0,9967 | 0,167 | 0,500 | 0,307 | 0,053 | 0,2567 | 0,1765 |
| Hist Gradient Boosting | 0,9925 | 0,9925 | 0,167 | 0,420 | 0,290 | 0,167 | 0,2608 | 0,1943 |
| Regressão Logística | 0,9908 | 0,9908 | 0,500 | 0,430 | 0,437 | 0,407 | 0,4433 | 0,3380 |

### Todas as estatísticas mais `rotation_hz` (65 features)

Fonte: [`comparison_all_with_hz.csv`](comparison_all_with_hz.csv). Executado incluindo `rotation_hz` entre as features.

| Modelo | 5-fold: acurácia | Rotação fora: acurácia média | Rotação fora: F1 macro |
|---|---|---|---|
| Random Forest | 0,9983 | 0,3358 | 0,2693 |
| Hist Gradient Boosting | 0,9933 | 0,3617 | 0,2542 |
| Regressão Logística | 0,9917 | 0,3342 | 0,2792 |

### Revocação por classe na rotação deixada de fora

Percentual de arquivos de cada classe classificados corretamente (calculado a partir das matrizes de confusão acumuladas; 200 arquivos por classe).

| Conjunto | Modelo | Baseline | Bearing | Bearing+Unbalance | Misalignment_NDE | Unbalance_DE | Unbalance_NDE |
|---|---|---|---|---|---|---|---|
| 64 features | Random Forest | 52 | 25 | 3 | 98 | 0 | 25 |
| 64 features | Hist Gradient Boosting | 50 | 38 | 20 | 89 | 0 | 22 |
| 64 features | Regressão Logística | 23 | 38 | 9 | 70 | 0 | 53 |
| 40 features (`shape`) | Random Forest | 58 | 25 | 0 | 66 | 0 | 6 |
| 40 features (`shape`) | Hist Gradient Boosting | 75 | 0 | 13 | 59 | 0 | 10 |
| 40 features (`shape`) | Regressão Logística | 90 | 46 | 66 | 40 | 0 | 25 |

### Principais observações

1. **Na validação cruzada os três modelos praticamente empatam em ~99%.** A diferença entre o melhor e o pior (99,83% e 99,17%) corresponde a 2 e 10 erros em 1.200 arquivos. Não há base para afirmar que um modelo é superior aqui.
2. **Quando a rotação de teste não aparece no treino, o desempenho cai para 26–44%**, bem acima do acaso (16,7%), mas muito abaixo dos 99%. As quedas ocorrem nos três modelos e nos três conjuntos de features.
3. **Com as 64 features, não há diferença clara entre os modelos na rotação deixada de fora** (acurácia média de 32% a 37%). O ranking muda de uma rotação para outra, e a variação de um mesmo modelo entre rotações (por exemplo, a Regressão Logística de 17% a 36%) é maior que a diferença entre modelos.
4. **Com as features de formato, a Regressão Logística se destaca** (44%, contra 26% dos dois modelos de árvores) e é a mais estável entre as rotações (41% a 50%). Random Forest e Gradient Boosting pioram em relação ao conjunto completo e chegam a 5% (Random Forest em 40 Hz) e 17% em 10 e 40 Hz, abaixo do acaso, o que indica erros sistemáticos e não aleatórios. Como são quatro rotações e uma única execução, a diferença é indicativa e não foi testada estatisticamente.
5. **As features de formato quase não reduzem a acurácia na validação cruzada** (99,1% a 99,7%, contra 99,2% a 99,8% com todas). A amplitude não é necessária para separar as classes quando as rotações já foram vistas.
6. **Incluir `rotation_hz` como feature não mudou o resultado** (acurácia média na rotação deixada de fora entre 33% e 36%, contra 32% a 37% sem ela). Isso é esperado: o valor da rotação testada nunca aparece no treino, e um modelo de árvores trata uma rotação inédita como a vizinha mais próxima que conhece.
7. **`Misalignment_NDE` é a única classe reconhecida de forma consistente entre rotações com as 64 features** (70% a 98% de revocação), e esse reconhecimento diminui com as features de formato (40% a 66%). Isso sugere que parte dessa assinatura está na intensidade do sinal, coerente com a EDA, que mostra energia maior nas bandas de 100 a 1.500 Hz nessa classe.
8. **`Unbalance_DE` nunca foi reconhecida na rotação deixada de fora** (0% em todos os modelos e conjuntos). Ela é confundida com `Unbalance_NDE` (os dois tipos de desbalanceamento, em lados diferentes do eixo) e com `Bearing+Unbalance`. Esses mesmos dois desbalanceamentos concentram os poucos erros da validação cruzada.
9. **Os erros na rotação deixada de fora aparecem em blocos de 50 arquivos**, ou seja, uma rotação inteira de uma classe é atribuída a outra de uma vez. Isso é compatível com os modelos reconhecendo grupos de arquivos parecidos (uma condição em uma rotação) e não uma assinatura da falha que valha para qualquer rotação.

### Interpretação

Os resultados mostram que a acurácia de ~99% da validação cruzada **mede o reconhecimento de condições e rotações já vistas, e não a capacidade de reconhecer a falha em uma velocidade nova**. Cada combinação de condição e rotação tem 50 arquivos muito semelhantes entre si, e o sorteio aleatório coloca arquivos parecidos no treino e no teste.

A hipótese inicial era que a queda se devesse à amplitude, que cresce com a rotação. Ela **só se sustenta em parte**: remover as features de amplitude melhorou a Regressão Logística, mas piorou as duas de árvores, e nenhum modelo recuperou o desempenho da validação cruzada. Uma explicação plausível, **não testada**, é que as frequências características das falhas são múltiplos da rotação e, portanto, mudam de posição quando ela muda; estatísticas no domínio do tempo, mesmo as de formato, não capturam isso de forma estável. Confirmar essa hipótese exige features no domínio da frequência normalizadas pela rotação.

## Limitações

- **Uma máquina, quatro rotações e 50 arquivos por condição e rotação.** O protocolo de rotação deixada de fora tem apenas quatro divisões, e duas delas (10 e 40 Hz) exigem extrapolação, por estarem fora da faixa de treino.
- **Possível proximidade entre arquivos de treino e teste na validação cruzada.** Os arquivos de uma mesma condição e rotação provavelmente foram gravados em sequência; não há informação no dataset para verificar isso. Se for o caso, os ~99% estão superestimados.
- **Uma única execução, com uma única semente (`random_state=42`).** Não há intervalos de confiança. Diferenças de poucos pontos entre modelos devem ser lidas como empate, principalmente na rotação deixada de fora, onde um mesmo modelo oscila 20 a 30 pontos entre rotações.
- **Hiperparâmetros padrão, sem ajuste**, por decisão do escopo: o objetivo era comparar famílias de modelos em igualdade de condições. Os resultados podem mudar com ajuste.
- **Só features no domínio do tempo no experimento inicial.** A comparação espectral adicional está documentada abaixo; ainda não foram testadas features de ordem sincronizadas pela rotação.
- **Linhas com valores ausentes são descartadas** (cerca de 0,3% das linhas), então as features de cada arquivo são calculadas sobre um pouco menos que os 36.000 pontos.
- **Licença do dataset não verificada.**

## Próximos passos

- **Features de ordem sincronizadas pela rotação:** analisar 1x, 2x, 3x e ordens superiores usando a referência angular do trigger. O experimento espectral atual usa bandas fixas em Hz e contrastes entre elas; ainda não alinha os espectros pelo giro do eixo.
- **Separar `Unbalance_DE` de `Unbalance_NDE`** com features calculadas por lado do eixo (por exemplo, razões entre os canais `de` e `nde`).
- **Cortar cada arquivo em janelas** (por exemplo, de 1 s), com agrupamento por arquivo na validação, para ter mais exemplos e medir a variação dentro de cada arquivo.
- **Repetir as avaliações com várias sementes** e relatar média e desvio, para saber quais diferenças entre modelos são reais.
- **Ajustar hiperparâmetros** em uma validação separada, caso o objetivo passe a ser escolher um modelo.
- **Verificar a ordem de gravação dos arquivos**, se estiver disponível, para checar a proximidade entre treino e teste na validação cruzada.

## Reprodução

```bash
python -m src.download_data
python -m src.build_dataset
python -m src.compare_models --features all
python -m src.compare_models --features shape
```

O experimento com `rotation_hz` foi obtido chamando `load_xy` com `drop_columns=()`.


## Experimento adicional: generalização com features espectrais

Este experimento complementa e preserva os resultados anteriores. Os CSVs históricos `comparison_all.csv`, `comparison_shape.csv` e `confusion_matrices.csv` não foram substituídos. Os resultados abaixo são escritos em arquivos separados pelo novo comando `python -m src.compare_generalization`.

**Nota de reprodução:** `data/processed/features.csv` não é versionado e não estava presente antes desta execução. Ao reconstruir as features temporais, o resultado `shape` coincidiu com o CSV histórico; no conjunto `all`, o Hist Gradient Boosting mudou de 36,50% para 36,17% e a Regressão Logística de 32,17% para 31,67% no protocolo B, enquanto o Random Forest permaneceu em 33,75%. A causa exata não pode ser estabelecida sem a tabela derivada usada na execução antiga. Os CSVs históricos foram mantidos; as comparações deste experimento novo usam uma única tabela gerada na mesma execução.

### Método

- O conjunto de features foi reconstruído a partir dos mesmos 1.200 arquivos e do mesmo leitor usado pelo modelo temporal. Assim, as features temporais e espectrais de cada linha usam exatamente os mesmos pontos finitos, permitindo uma comparação controlada.
- As features espectrais são RMS de energia em quatro bandas já exploradas na EDA: 0–100, 100–500, 500–1.500 e 1.500–3.000 Hz, calculadas com periodograma Hann a 6 kHz. Foram usadas apenas as quatro acelerações: o registro do dataset informa que os canais de velocidade são integrados a partir da aceleração ([Zenodo](https://zenodo.org/records/14388093)).
- O conjunto reduzido de 32 features mantém RMS, pico a pico, assimetria e curtose nos oito canais. Remove média, desvio padrão, média absoluta e pico, que são redundantes ou muito próximos de outras medidas neste sinal.
- O conjunto espectral ILR tem 12 features: três contrastes logarítmicos independentes entre as quatro bandas em cada acelerômetro. Isso evita usar quatro proporções que somam 1 e, portanto, têm uma dependência matemática.
- Não foi feita busca de hiperparâmetros. Os seis conjuntos fixos são comparados sem ajuste; além disso, um seletor aninhado compara os conjuntos por GroupKFold somente nas rotações de treino, escolhe pela F1 macro interna e avalia depois na rotação externa. A rotação não entra como preditora: serve para separar os grupos. A padronização da Regressão Logística permanece dentro do pipeline de cada fold.

O seletor aninhado escolheu o conjunto espectral ILR de 12 features para os três modelos em todas as quatro rotações externas. As métricas resultantes coincidem com a linha ILR da tabela. As escolhas por divisão estão em `docs/selected_features_by_rotation.csv`.

| Modelo | Features escolhidas nas quatro divisões | Acurácia externa média | F1 macro médio por rotação |
|---|---|---:|---:|
| Random Forest | ILR espectral (12) | 67,00% | 0,584 |
| Hist Gradient Boosting | ILR espectral (12) | 60,67% | 0,541 |
| Regressão Logística | ILR espectral (12) | 65,50% | 0,599 |

### Resultados do protocolo B — rotação deixada de fora

Cada célula mostra acurácia / F1 macro médio entre as quatro rotações deixadas de fora. O F1 dessa tabela é a média dos quatro resultados por rotação; ele pode diferir do F1 calculado uma única vez sobre as previsões agregadas.

| Features | Random Forest | Hist Gradient Boosting | Regressão Logística |
|---|---:|---:|---:|
| 64 temporais | 33,75% / 0,245 | 36,17% / 0,255 | 31,67% / 0,265 |
| 40 de formato | 25,67% / 0,177 | 26,08% / 0,194 | 44,33% / 0,338 |
| 32 temporais reduzidas | 40,75% / 0,307 | 36,33% / 0,260 | 44,58% / 0,380 |
| 16 espectrais absolutas | 37,08% / 0,254 | 33,92% / 0,221 | **67,17% / 0,589** |
| 12 espectrais ILR | **67,00% / 0,584** | **60,67% / 0,541** | 65,50% / **0,599** |
| Formato + espectro (56) | 43,83% / 0,329 | 38,75% / 0,285 | 57,67% / 0,470 |

A melhor representação muda conforme o algoritmo. O conjunto ILR de 12 features melhora os três modelos em acurácia média comparado ao melhor resultado antigo de cada família. Com as 16 bandas absolutas, a Regressão Logística tem a maior acurácia e F1 médio observados. Isso é evidência promissora, mas não prova que esse conjunto será o melhor em novas máquinas ou em velocidades fora desta base.

### Resultados do protocolo A — 5-fold

A acurácia média continuou alta para a maioria das combinações, porque as quatro velocidades aparecem no treino e no teste. Para as 16 features espectrais absolutas, a Regressão Logística ficou em 94,08%; para as 12 ILR, os três modelos ficaram entre 96,33% e 99,25%. Esses números continuam respondendo à avaliação em velocidades já representadas no treino.

| Features | Random Forest | Hist Gradient Boosting | Regressão Logística |
|---|---:|---:|---:|
| 64 temporais | 99,83% | 99,33% | 99,17% |
| 40 de formato | 99,67% | 99,25% | 99,08% |
| 32 temporais reduzidas | 99,75% | 99,33% | 98,42% |
| 16 espectrais absolutas | 99,25% | 99,25% | 94,08% |
| 12 espectrais ILR | 99,25% | 98,67% | 96,33% |
| Formato + espectro (56) | 99,75% | 99,42% | 99,42% |

### Velocidade e desempenho por classe

A Regressão Logística com 16 features espectrais absolutas obteve 50,0% ao deixar 10 Hz de fora, 70,0% em 20 Hz, 84,0% em 30 Hz e 64,67% em 40 Hz. O resultado não cai abaixo do acaso em nenhuma rotação, mas a variação entre velocidades ainda é relevante.

Na matriz agregada das quatro rotações, a revocação por classe dessa configuração foi:

| Classe | Revocação |
|---|---:|
| Baseline | 99,5% |
| Bearing | 75,0% |
| Bearing+Unbalance | 50,0% |
| Misalignment_NDE | 75,0% |
| Unbalance_DE | 75,0% |
| Unbalance_NDE | 28,5% |

Isso melhora o problema anterior de `Unbalance_DE`, que tinha revocação de 0% nos experimentos temporais antigos. Porém, `Unbalance_NDE` continua difícil. A acurácia não deve ser interpretada sem a matriz e as métricas por classe. No conjunto ILR com Random Forest, por exemplo, a acurácia é 67%, mas `Bearing+Unbalance` quase nunca é reconhecida; por isso essa configuração não é recomendada apenas com base na acurácia.

Para a seleção aninhada, a revocação pooled por classe é:

| Classe | Random Forest + ILR | Hist Gradient Boosting + ILR | Regressão Logística + ILR |
|---|---:|---:|---:|
| Baseline | 100,0% | 84,0% | 100,0% |
| Bearing | 99,5% | 46,0% | 61,5% |
| Bearing+Unbalance | 0,5% | 36,0% | 25,0% |
| Misalignment_NDE | 100,0% | 95,5% | 100,0% |
| Unbalance_DE | 64,0% | 68,5% | 78,5% |
| Unbalance_NDE | 38,0% | 34,0% | 28,0% |

O CSV de comparação traz métricas macro por fold e também uma linha `pooled`, calculada sobre todas as previsões agregadas. Como F1 macro é não linear, a média dos F1 por velocidade não é igual ao F1 da matriz agregada. O dashboard e o relatório distinguem essas duas medidas.

### Interpretação e limites

- O ganho aponta para a distribuição de energia por frequência como uma descrição mais útil para a generalização entre rotações que as estatísticas temporais isoladas. É uma pista consistente com a EDA, não uma demonstração de uma assinatura mecânica universal.
- Os novos resultados são comparações exploratórias nos mesmos quatro grupos de rotação já utilizados no desenvolvimento. Como examinamos esses grupos para encontrar boas representações, o melhor valor observado não é um teste externo independente. Para uma estimativa confirmatória, seria necessário avaliar uma nova velocidade, uma nova máquina ou uma aquisição independente.
- Não há metadados de sessão suficientes para formar grupos de aquisição confiáveis; nenhum grupo de sessão foi inventado.
- As rotações de 10 e 40 Hz ficam fora do intervalo usado para treinar em cada uma dessas divisões; 20 e 30 Hz ficam entre velocidades de treino. Por isso os quatro resultados devem permanecer visíveis, e não só a média.
- Embora a rotina aninhada não consulte a rotação externa para selecionar as features, durante este trabalho já examinamos resultados exploratórios nos mesmos quatro grupos ao formular os candidatos espectrais. Portanto, os valores atuais são evidência de desenvolvimento, não um teste final totalmente cego. Uma nova aquisição, outra máquina ou outra velocidade é necessária para confirmação independente.
- Os modelos e hiperparâmetros originais foram preservados. Não afirmamos que o maior número encontrado seja uma garantia de desempenho operacional.
- O 5-fold continua sendo uma avaliação de medições em rotações conhecidas; não substitui o protocolo de rotação deixada de fora.

### Arquivos novos e reprodução

- `src/features.py`: extração do RMS espectral nas bandas usadas pela EDA.
- `src/build_dataset.py`: opção para gerar uma tabela espectral separada, sem sobrescrever `features.csv`.
- `src/compare_generalization.py`: comparação fixa dos seis conjuntos nos dois protocolos, com métricas por classe e matrizes.
- `docs/comparison_generalization.csv`: resultados por modelo, protocolo e divisão.
- `docs/class_metrics_generalization.csv`: precisão, revocação, F1 e suporte por classe.
- `docs/confusion_matrices_generalization.csv`: matrizes agregadas e por rotação.
- `docs/selected_features_by_rotation.csv`: conjunto escolhido dentro de cada treino externo.
- `app/results_dashboard.py`: exibe os novos conjuntos, protocolos, matrizes e métricas por classe.

Reproduza a tabela espectral e os resultados com:

```bash
python -m src.build_dataset --with-spectral
python -m src.compare_generalization
streamlit run app/results_dashboard.py
```
