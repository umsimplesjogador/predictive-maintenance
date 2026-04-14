import requests
import json
import pandas as pd

# Montamos a URL do seu endpoint POST na nuvem
URL = "https://predictive-maintenance-jz9x.onrender.com/predict"

def test_cloud_prediction():
    print("Carregando dados locais para simular o equipamento...")
    df = pd.read_csv("Test-O_G_Equipment_Data.csv")
    df['Fail'] = df['Fail'].fillna(0).astype(int)
    
    df['Target'] = df['Fail'].shift(-5).rolling(5).max().fillna(0)
    predict_df = df[df['Fail'] == 0].copy()
    
    predict_df['Temperature_roll_mean_3'] = predict_df['Temperature'].shift(1).rolling(3).mean().fillna(predict_df['Temperature'])
    predict_df['Temperature_roll_std_3'] = predict_df['Temperature'].shift(1).rolling(3).std().fillna(0)
    predict_df['Temperature_diff'] = predict_df['Temperature'].diff().fillna(0)
    
    predict_df['Pressure_roll_mean_3'] = predict_df['Pressure'].shift(1).rolling(3).mean().fillna(predict_df['Pressure'])
    predict_df['Pressure_roll_std_3'] = predict_df['Pressure'].shift(1).rolling(3).std().fillna(0)
    predict_df['Pressure_diff'] = predict_df['Pressure'].diff().fillna(0)

    predict_df['VibrationX_roll_mean_3'] = predict_df['VibrationX'].shift(1).rolling(3).mean().fillna(predict_df['VibrationX'])
    predict_df['VibrationX_roll_std_3'] = predict_df['VibrationX'].shift(1).rolling(3).std().fillna(0)
    predict_df['VibrationX_diff'] = predict_df['VibrationX'].diff().fillna(0)
    
    predict_df['VibrationY_roll_mean_3'] = predict_df['VibrationY'].shift(1).rolling(3).mean().fillna(predict_df['VibrationY'])
    predict_df['VibrationY_roll_std_3'] = predict_df['VibrationY'].shift(1).rolling(3).std().fillna(0)
    predict_df['VibrationY_diff'] = predict_df['VibrationY'].diff().fillna(0)
    
    predict_df['VibrationZ_roll_mean_3'] = predict_df['VibrationZ'].shift(1).rolling(3).mean().fillna(predict_df['VibrationZ'])
    predict_df['VibrationZ_roll_std_3'] = predict_df['VibrationZ'].shift(1).rolling(3).std().fillna(0)
    predict_df['VibrationZ_diff'] = predict_df['VibrationZ'].diff().fillna(0)
    
    predict_df['Frequency_roll_mean_3'] = predict_df['Frequency'].shift(1).rolling(3).mean().fillna(predict_df['Frequency'])
    predict_df['Frequency_roll_std_3'] = predict_df['Frequency'].shift(1).rolling(3).std().fillna(0)
    predict_df['Frequency_diff'] = predict_df['Frequency'].diff().fillna(0)
    
    # Amostra Saudável
    sample_normal = dict(predict_df.iloc[1])
    
    # Amostra onde sabemos que estava perto de falhar
    sample_danger = dict(predict_df.iloc[505]) # Ponto cirúrgico onde o modelo encontrou falha térmica
    
    # Deletando colunas que a API não consome
    for sample in [sample_normal, sample_danger]:
        for col in ['Cycle', 'Fail', 'Target', 'Failure_Event']:
            sample.pop(col, None)

    print("\nDisparando [Teste 1] - Equipamento Saudável para a Nuvem de SP/EUA...")
    res_normal = requests.post(URL, json=sample_normal)
    print("Resposta da Nuvem:")
    print(json.dumps(res_normal.json(), indent=2, ensure_ascii=False))
    
    print("\nDisparando [Teste 2] - Equipamento Degradando (Risco) para a Nuvem...")
    res_danger = requests.post(URL, json=sample_danger)
    print("Resposta da Nuvem:")
    print(json.dumps(res_danger.json(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_cloud_prediction()
