# Relatório Preditivo de Falhas (Análise Sênior)

Este relatório técnico sumariza a mecânica interna do modelo e o perfil inercial do maquinário utilizando estatística paramétrica profunda. Destina-se às equipes de Manutenção Operacional e Engenheiros Sêniores de Dados.

## 1. Horizonte Preditivo Instituído
O modelo não infere se a máquina "está falhando no momento" (uma inferência inútil para preditividade real). O _Target_ foi modelado com a política de _Lead Time_ fixada: **"Haverá um evento de falha nos próximos 5 ciclos de acompanhamento?"**. 
Essa defasagem exige rigorosa blindagem de *Data Leakage*, sendo a base fatiada no modelo "Out-of-Time" (treinamento restrito ao passado contíguo).

## 2. Interpretações de Causa-Raiz (SHAP Value Findings e Correlações de Pearson)

### A. O Gatilho Absoluto: Vibração (Eixo Y)
O pacote de interpretabilidade `SHAP` e as matrizes de Pearson atestaram correlações fortíssimas:
- **Correlação Linear com a Falha**: `0.455` (Sendo a métrica vital isolada mais expressiva de toda engenharia do sistema).
- **Mapeamento Clínico**: Em operação normal, a média de vibração Y sustenta-se confortavelmente em **~68**. Diante de falhas iminentes, esse eixo esgarça numa turbulência para a média de **~122** (um salto contínuo violento muito acima de desvios-padrão usuais).
- O modelo aprendeu que a *Variação (Slope)* deste sensor é o que dita a catástrofe preditiva, e não apenas números fixos em abismo.

### B. O Fator Térmico: Temperatura como "Slope de Confirmação"
Enquanto `VibrationY` causa as pontadas, a Temperatura entra em fervura constante confirmando o processo final rumo à quebra:
- **Taxa de Fervura Calculada (Slope):** Isolando os exatos 5 últimos ciclos das falhas fatais, aplicamos regressão inercial para diagnosticar a curvatura de exaustão. A temperatura engole o metal com uma taxa confirmada de **+ 6.4°C a cada pulso/ciclo seguinte**.
- Sensores captando inclinações contíguas nessa agressividade nos 3 últimos ciclos acionam uma sobreposição fatal nos Thresholds do Algoritmo LightGBM em produção endossando a Falha Rápida.

## 3. Matriz de Mortalidade de Presets
A combinação calibral das matrizes dita a expectativa de vida da máquina brutalmente.
Uma análise cruzada do sistema atestou o agrupamento anômalo. O limite de stress fatal foi mapeado estatisticamente e atestou que uma máquina correndo sob `Preset 1 (Nível 1)` e simultaneamente no `Preset 2 (Nível 5)` atinge absurdos **16.1% de mortalidade geral** no pipeline. Essa calibração mecânica cruzada está condenada no algoritmo e sua mitigação urgente é recomendada ao departamento C-Level.
