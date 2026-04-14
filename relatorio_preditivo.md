# Relatório Preditivo de Falhas

Este relatório técnico objetiva sumarizar a mecânica interna do modelo e as condições subjacentes que regem as previsões do motor `XGBoost` construído. O relatório destina-se às equipes de engenharia de ciclo (Manutenção Operacional).

## Horizonte Preditivo Instituído
O modelo não infere se a máquina "está falhando no momento" (uma inferência inútil para preditividade real). O _Target_ de negócio foi modelado com a política de _Lead Time_ fixada em: **"Haverá um evento de falha nos próximos 5 ciclos de acompanhamento?"**. 
Essa defasagem assegura que haja tempo hábil para agendamento de uma intervenção corretiva ou freio planejado no uso do equipamento. Delinear este formato exige o isolamento perfeito dos dados _Time-Based Split_, para não haver contaminação do "futuro para o passado" nos treinos temporais.

## Interpretações de Causa-Raiz (SHAP Value Findings)

O pacote de interpretabilidade `SHAP` identificou os fatores primários que as árvores de decisão utilizam nos bastidores para "apitar um alarme". 

1. **Variância de Vibração Y (VibrationY_roll_std)**: Constatou-se não só o valor absoluto elevado, mas o desvio repentino do histórico curto é o preditor com maior magnitude da dor mecânica que acarreta na falha.
2. **Pressão Anômala Recente**: Pressão elevada consistentemente detectada fora dos intervalos previstos na baseline. 
3. **Sensores Terciários**: Temperatura e Frequência atuam como *features de composição*. Altas temperaturas isoladamente são falsos positivos, porém, quando associadas a anomalias de vibração do Eixo Y configuram forte embasamento probalístico na falha do ciclo próximo.

## Tendência Praticada *Antes* do Fato (Degradação)

A média dos dados na janela de 1 a 3 ciclos imediatamente anteriores ao colapso mecânico demonstrou padrões muito claros:
* A temperatura cresce drasticamente, passando em média de 67°C (estado sadio) para quase 90°C.
* A Vibração no eixo Y é o alerta central (saltando em média de 68 para algo próximo a 111 de amplitude vibracional).
* **Conclusão Comportamental:** O equipamento não rompe subitamente; ele entra num quadro de aceleração de degeneração cerca de 3 a 4 ciclos que precedem o colapso, criando esta janela proeminente de atuação preditiva do algoritmo.
