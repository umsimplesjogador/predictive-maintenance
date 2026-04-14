# 🏭 Avaliação de Engenharia Preditiva: Manutenção O&G

Bem-vindo ao repositório oficial do Sistema Inteligente de Manutenção Preditiva (Predictive Maintenance O&G).
Este projeto implanta IA Sênior contumaz usando dados sensoriais contínuos para inferir e alertar abates preditivos de equipamentos nos pórticos mecânicos. 

---

## 🏗 Arquitetura
Este projeto segue rigorosamente parâmetros de Clean-Architecture acoplados à modelagens de Custo Sensitivo sobre séries temporais extremas, sendo desenhado aos moldes do processo seletivo (*Shape Digital*). O backend é consumido pela filial `main` blindada que alimenta diretamente Nuvem de Deploy via conteinerização enxuta.

- **`src/`:** Morada de desenvolvimento python de Inteligência (`data_preprocessing_and_modeling.py`) rodando pipeline Tunado em LightGBM/Optuna/MLFlow, bem como o emulador de consumo Cloud (`test_api.py`).
- **`docs/`:** Diretório de Manuais, C-Level Pitchs, Documentação Preditiva e Guias de Defesa Científica e Entrevista.
- **`notebooks/`:** A EDA Notebook Storytelling com plotagens massivas e profundidade em regressão linear e análises paramétricas de cruzamento.
- **`Root/`:** Apenas o ambiente estrito do Payload Cloud (`app.py`, `Dockerfile` `xgboost_model.pkl`), desenhado intencionalmente para não inflar e poluir a imagem Docker a ser Buildada pelos WebServices.

## 🚀 Como acionar e Testar a Nuvem

O serviço é servido no núcleo REST na Nuvem, programado em **FastAPI** rodando num contêiner Linux no Render.

### Testes Manuais Visuais:
Acesse abertamente a área de Homologação em SwaggerUI da nossa IA online neste exato link:
👉 **[Documentação de Consumo Swagger (Nuvem em Temp-Real)](https://predictive-maintenance-jz9x.onrender.com/docs)**

1. Expanda a janela amarela de rota `/predict`.
2. Clique no botão de canto `Try it Out`.
3. Popule livremente o Input JSON com telemetria presuntiva (Por exemplo: infle o Vibration_X, jogue Preset1 e 2 para margens insalubres e jogue a Temperature lá em cima).
4. Clique em Executar e presencie a nuvem atirar de volta um "1": "Warning: Probable Failure".

### Testes Programáticos:
Basta acionar localmente de sua máquina a suíte script do repositório:
```bash
python src/test_api.py
```
Essa rotina buscará uma temperatura de quebra cirúrgica pré mapeada em nossos bancos locais sob aproximação histórica, fabricará os pacotes e extrairá em Milisegundos a previsão real provida peló Node dos EUA.

## 🛠️ Tecnologias de Elite no Ecossistema
⚙️ `LightGBM & XGBoost` (Classificação Estocástica)\
🧪 `Optuna & MLflow` (Tracking local Paramétrico e Hiper Otimização Autônoma)\
🧠 `SHAP.TreeExplainer` (Interpretatividade do Motor Causal em Boxplots Inerentes)\
🧰 `FastAPI + Uvicorn` (Asymmetric Core Frameworks App)\
🐳 `Docker Hub (Python-Slim + Libgomp1)` (Encapsulador nativo para compatibilidade C++ dos Modelos)\
🌐 `Render Cloud + Github Webhooks` (Trigger Master Deploy CI/CD)

---
*Assinado: Fabio Geliustique da Silva | Senior Data Scientist*
