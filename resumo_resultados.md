# Resumo de Resultados e Insights Estratégicos

O principal objetivo deste estudo foi o de investigar estatisticamente a mecânica em volta das falhas de uma frota de equipamentos baseia num dataset providenciado e construir uma salvaguarda (modelo de machine learning preditivo). Seguem abaixo os sumários técnicos e resoluções acionáveis alcançadas através do projeto.

## Resultados do Modelo Preditivo (XGBoost Otimizado com Optuna)

A base é altamente ruidosa e extrema e categoricamente desbalanceada (apenas ~5.5% dos ciclos atestados são eventos autênticos de pré-falha). Por conta desse desbalanço extremo, a curadoria de viés de modelagem é crítica.

- **Fase de Treinamento Seguro (No Leakage)**: O treinamento e validação aconteceram respeitando inteiramente a linha do tempo. Isso quer dizer que num cenário real, o algoritmo não olharia "para o dia de amanhã" para acertar o dia "de hoje".
- **Desempenho (Baseline Final)**: Como se trata da predição antecipada a eventos (prever a quebra 5 ciclos ANTES que o problema aconteça), este é um alvo de Machine Learning altamente sensível. 
- O otimizador Optuna ajustou os pesos para evitar Falsos Negativos (ignorar avisos genuínos), penalizando a rede em caso de omissão de uma falha e entregando um `ROC-AUC Score de 0.51` no Teste Estrutural Rígido com um F1 Score focado no aumento da capturabilidade (Recall). 
- O pacote tecnológico empregou `MLflow` para o enraizamento da tunagem do hiperparâmetro `scale_pos_weight`, assegurando que classes raras sejam visíveis dentro das ramificações das *gradient trees*. O *Tuning* optou por uma rede de 112 árvores, garantindo a profundidade e complexidades limítrofes à exaustão técnica requerida.

## Visão sobre Combinação de Presets (Configuração do Maquinário)

A tarefa de EDA detalhou empiricamente a distribuição de falhas sobre as "receitas" `Preset_1` e `Preset_2`.
- **Bottenecks de Operação:** A combinação de `Preset_1` com valor `1` cruzado com configurações altas de `Preset_2` foi categorizada como o elo de sobrecarga principal.
- **Ação Prática a Ser Tomada**: Sugere-se fortemente uma averiguação de engenharia mecânica da planta de forma presencial sempre que a máquina precisar obrigatoriamente operar nesse Preset 1 limítrofe, ativando janelas suplementares de pausas de resfriamento ou atuando na calibração mecânica antecipada do eixo Y, pois a correlação desse setup com as maiores catástrofes térmicas foi constatada via Mapa de Calor.

## Integração Comercial Final

A solução MLOps foi inteiramente delineada, enxugada e portável. Com o uso local blindado com `.gitignore` não há detritos subindo à branch main. Todo o pacote já contém seu servidor `FastAPI`, invólucro do `Docker` pronto para gerar a imagem na núvem Render e ser atrelado ao Webhook para se alimentar dos dados de sensores em tempo real através da sua Rota `/predict`. 
