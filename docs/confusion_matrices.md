# Matrizes de confusão — rotação deixada de fora

Matrizes de confusão dos três modelos no protocolo **rotação deixada de fora** (treina em três rotações e testa na quarta, repetindo para 10, 20, 30 e 40 Hz). Cada matriz **soma as quatro rotações testadas**: cada classe tem 200 arquivos (50 por rotação), então cada linha soma 200 e o total é 1.200.

- **Linhas:** classe real. **Colunas:** classe prevista. A diagonal são os acertos.
- **Células:** contagem de arquivos e, entre parênteses, o percentual da linha (revocação, no caso da diagonal).
- **Dados em formato de tabela** (para dashboards): [`confusion_matrices.csv`](confusion_matrices.csv).
- **Origem:** saída de `python -m src.compare_models`. A acurácia calculada pela diagonal de cada matriz confere com `leave_out_accuracy` em `comparison_*.csv`.
- **Não incluído:** matrizes da validação cruzada de 5 partes e matrizes separadas por rotação; o script atual só guarda a matriz acumulada da rotação deixada de fora.

## 64 features (todas as estatísticas), sem `rotation_hz`

Métricas gerais em [`comparison_all.csv`](comparison_all.csv).

### Random Forest (acurácia 33.8%)

| real \ previsto | Baseline | Bearing | Bearing+Unb. | Misalign_NDE | Unbal_DE | Unbal_NDE |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | **104 (52%)** | 49 (25%) | 1 (1%) | 46 (23%) | 0 (0%) | 0 (0%) |
| Bearing | 0 (0%) | **50 (25%)** | 50 (25%) | 50 (25%) | 50 (25%) | 0 (0%) |
| Bearing+Unbalance | 50 (25%) | 95 (48%) | **5 (3%)** | 50 (25%) | 0 (0%) | 0 (0%) |
| Misalignment_NDE | 0 (0%) | 4 (2%) | 0 (0%) | **196 (98%)** | 0 (0%) | 0 (0%) |
| Unbalance_DE | 87 (44%) | 73 (37%) | 0 (0%) | 40 (20%) | **0 (0%)** | 0 (0%) |
| Unbalance_NDE | 50 (25%) | 50 (25%) | 0 (0%) | 50 (25%) | 0 (0%) | **50 (25%)** |

### Hist Gradient Boosting (acurácia 36.5%)

| real \ previsto | Baseline | Bearing | Bearing+Unb. | Misalign_NDE | Unbal_DE | Unbal_NDE |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | **100 (50%)** | 41 (21%) | 45 (23%) | 0 (0%) | 9 (5%) | 5 (3%) |
| Bearing | 0 (0%) | **76 (38%)** | 50 (25%) | 24 (12%) | 50 (25%) | 0 (0%) |
| Bearing+Unbalance | 60 (30%) | 50 (25%) | **40 (20%)** | 50 (25%) | 0 (0%) | 0 (0%) |
| Misalignment_NDE | 0 (0%) | 21 (11%) | 0 (0%) | **178 (89%)** | 0 (0%) | 1 (1%) |
| Unbalance_DE | 56 (28%) | 29 (15%) | 61 (31%) | 50 (25%) | **0 (0%)** | 4 (2%) |
| Unbalance_NDE | 56 (28%) | 50 (25%) | 0 (0%) | 50 (25%) | 0 (0%) | **44 (22%)** |

### Logistic Regression (acurácia 32.2%)

| real \ previsto | Baseline | Bearing | Bearing+Unb. | Misalign_NDE | Unbal_DE | Unbal_NDE |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | **46 (23%)** | 29 (15%) | 79 (40%) | 25 (13%) | 0 (0%) | 21 (11%) |
| Bearing | 0 (0%) | **76 (38%)** | 100 (50%) | 24 (12%) | 0 (0%) | 0 (0%) |
| Bearing+Unbalance | 49 (25%) | 83 (42%) | **18 (9%)** | 0 (0%) | 0 (0%) | 50 (25%) |
| Misalignment_NDE | 0 (0%) | 0 (0%) | 7 (4%) | **140 (70%)** | 10 (5%) | 43 (22%) |
| Unbalance_DE | 0 (0%) | 0 (0%) | 97 (49%) | 0 (0%) | **0 (0%)** | 103 (52%) |
| Unbalance_NDE | 0 (0%) | 0 (0%) | 94 (47%) | 0 (0%) | 0 (0%) | **106 (53%)** |

## 40 features de formato (`shape`), sem `rotation_hz`

Métricas gerais em [`comparison_shape.csv`](comparison_shape.csv).

### Random Forest (acurácia 25.7%)

| real \ previsto | Baseline | Bearing | Bearing+Unb. | Misalign_NDE | Unbal_DE | Unbal_NDE |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | **116 (58%)** | 0 (0%) | 0 (0%) | 22 (11%) | 34 (17%) | 28 (14%) |
| Bearing | 0 (0%) | **50 (25%)** | 100 (50%) | 50 (25%) | 0 (0%) | 0 (0%) |
| Bearing+Unbalance | 88 (44%) | 53 (27%) | **0 (0%)** | 50 (25%) | 6 (3%) | 3 (2%) |
| Misalignment_NDE | 0 (0%) | 6 (3%) | 13 (7%) | **131 (66%)** | 50 (25%) | 0 (0%) |
| Unbalance_DE | 63 (32%) | 0 (0%) | 50 (25%) | 87 (44%) | **0 (0%)** | 0 (0%) |
| Unbalance_NDE | 34 (17%) | 18 (9%) | 32 (16%) | 100 (50%) | 5 (3%) | **11 (6%)** |

### Hist Gradient Boosting (acurácia 26.1%)

| real \ previsto | Baseline | Bearing | Bearing+Unb. | Misalign_NDE | Unbal_DE | Unbal_NDE |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | **150 (75%)** | 0 (0%) | 20 (10%) | 0 (0%) | 0 (0%) | 30 (15%) |
| Bearing | 0 (0%) | **0 (0%)** | 11 (6%) | 123 (62%) | 57 (29%) | 9 (5%) |
| Bearing+Unbalance | 50 (25%) | 0 (0%) | **26 (13%)** | 50 (25%) | 74 (37%) | 0 (0%) |
| Misalignment_NDE | 4 (2%) | 29 (15%) | 12 (6%) | **117 (59%)** | 38 (19%) | 0 (0%) |
| Unbalance_DE | 50 (25%) | 2 (1%) | 15 (8%) | 133 (67%) | **0 (0%)** | 0 (0%) |
| Unbalance_NDE | 30 (15%) | 50 (25%) | 0 (0%) | 100 (50%) | 0 (0%) | **20 (10%)** |

### Logistic Regression (acurácia 44.3%)

| real \ previsto | Baseline | Bearing | Bearing+Unb. | Misalign_NDE | Unbal_DE | Unbal_NDE |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | **180 (90%)** | 0 (0%) | 20 (10%) | 0 (0%) | 0 (0%) | 0 (0%) |
| Bearing | 0 (0%) | **92 (46%)** | 108 (54%) | 0 (0%) | 0 (0%) | 0 (0%) |
| Bearing+Unbalance | 0 (0%) | 69 (35%) | **131 (66%)** | 0 (0%) | 0 (0%) | 0 (0%) |
| Misalignment_NDE | 0 (0%) | 49 (25%) | 22 (11%) | **79 (40%)** | 0 (0%) | 50 (25%) |
| Unbalance_DE | 0 (0%) | 0 (0%) | 100 (50%) | 49 (25%) | **0 (0%)** | 51 (26%) |
| Unbalance_NDE | 0 (0%) | 49 (25%) | 1 (1%) | 100 (50%) | 0 (0%) | **50 (25%)** |

## 64 features + `rotation_hz` (65 colunas)

Métricas gerais em [`comparison_all_with_hz.csv`](comparison_all_with_hz.csv).

### Random Forest (acurácia 33.6%)

| real \ previsto | Baseline | Bearing | Bearing+Unb. | Misalign_NDE | Unbal_DE | Unbal_NDE |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | **106 (53%)** | 50 (25%) | 0 (0%) | 44 (22%) | 0 (0%) | 0 (0%) |
| Bearing | 0 (0%) | **64 (32%)** | 49 (25%) | 50 (25%) | 37 (19%) | 0 (0%) |
| Bearing+Unbalance | 50 (25%) | 98 (49%) | **2 (1%)** | 50 (25%) | 0 (0%) | 0 (0%) |
| Misalignment_NDE | 0 (0%) | 15 (8%) | 0 (0%) | **185 (93%)** | 0 (0%) | 0 (0%) |
| Unbalance_DE | 77 (39%) | 49 (25%) | 34 (17%) | 40 (20%) | **0 (0%)** | 0 (0%) |
| Unbalance_NDE | 7 (4%) | 44 (22%) | 56 (28%) | 47 (24%) | 0 (0%) | **46 (23%)** |

### Hist Gradient Boosting (acurácia 36.2%)

| real \ previsto | Baseline | Bearing | Bearing+Unb. | Misalign_NDE | Unbal_DE | Unbal_NDE |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | **100 (50%)** | 42 (21%) | 45 (23%) | 0 (0%) | 8 (4%) | 5 (3%) |
| Bearing | 0 (0%) | **76 (38%)** | 50 (25%) | 24 (12%) | 50 (25%) | 0 (0%) |
| Bearing+Unbalance | 60 (30%) | 50 (25%) | **40 (20%)** | 50 (25%) | 0 (0%) | 0 (0%) |
| Misalignment_NDE | 0 (0%) | 25 (13%) | 0 (0%) | **174 (87%)** | 0 (0%) | 1 (1%) |
| Unbalance_DE | 56 (28%) | 60 (30%) | 48 (24%) | 32 (16%) | **0 (0%)** | 4 (2%) |
| Unbalance_NDE | 56 (28%) | 54 (27%) | 0 (0%) | 46 (23%) | 0 (0%) | **44 (22%)** |

### Logistic Regression (acurácia 33.4%)

| real \ previsto | Baseline | Bearing | Bearing+Unb. | Misalign_NDE | Unbal_DE | Unbal_NDE |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | **46 (23%)** | 30 (15%) | 75 (38%) | 24 (12%) | 0 (0%) | 25 (13%) |
| Bearing | 0 (0%) | **81 (41%)** | 100 (50%) | 19 (10%) | 0 (0%) | 0 (0%) |
| Bearing+Unbalance | 49 (25%) | 82 (41%) | **18 (9%)** | 0 (0%) | 1 (1%) | 50 (25%) |
| Misalignment_NDE | 0 (0%) | 0 (0%) | 8 (4%) | **148 (74%)** | 1 (1%) | 43 (22%) |
| Unbalance_DE | 0 (0%) | 0 (0%) | 97 (49%) | 0 (0%) | **0 (0%)** | 103 (52%) |
| Unbalance_NDE | 0 (0%) | 0 (0%) | 92 (46%) | 0 (0%) | 0 (0%) | **108 (54%)** |
