import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configurações visuais avançadas
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Fail'] = df['Fail'].fillna(0).astype('int')
    return df

def plot_distributions(df, vars_to_analyze):
    """Cria Violin plots comparando estados normais vs falha"""
    plt.figure(figsize=(15, 10))
    for i, var in enumerate(vars_to_analyze, 1):
        plt.subplot(2, 3, i)
        sns.violinplot(data=df, x='Fail', y=var, hue='Fail', palette={0: "#2ecc71", 1: "#e74c3c"}, legend=False)
        plt.title(f'Distribuição de {var}')
        plt.xlabel('Status (0 = Normal, 1 = Falha)')
    plt.tight_layout()
    plt.savefig('violin_plots_falhas.png')
    plt.close()

def plot_pairplots(df, vars_to_analyze):
    """Cria Scatter Plots entre as variáveis destacando as falhas"""
    g = sns.pairplot(df[vars_to_analyze + ['Fail']], hue='Fail', palette={0: "#3498db", 1: "#e74c3c"}, 
                     diag_kind="kde", corner=True, plot_kws={'alpha':0.6})
    g.fig.suptitle('Pairplot de Variáveis Contínuas (Normal vs Falha)', y=1.02)
    plt.savefig('pairplot_falhas.png')
    plt.close()

def plot_preset_comparison(df, vars_to_analyze):
    """Compara o comportamento pelas configurações de Preset"""
    plt.figure(figsize=(15, 10))
    for i, var in enumerate(vars_to_analyze, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(data=df, x='Preset_1', y=var, hue='Fail', palette={0: "#95a5a6", 1: "#c0392b"})
        plt.title(f'{var} por Preset_1')
    plt.tight_layout()
    plt.savefig('boxplot_presets.png')
    plt.close()
    
def feature_engineering_anomalies(df, vars_to_analyze):
    """Calcula estatísticas móveis para detectar anomalias pré-falha"""
    df_sorted = df.copy() 
    print("=== Relatório Estatístico Simplificado ===")
    
    print("\nMédias no momento da falha vs Normal:")
    print(df_sorted.groupby('Fail')[vars_to_analyze].mean())
    
    # Capturar instantes (índices) prévios às falhas (1 a 3 ciclos antes)
    fail_idx = df_sorted[df_sorted['Fail'] == 1].index
    pre_fail_data = []
    
    for idx in fail_idx:
        start = max(0, idx-3)
        if start < idx:
            pre_fail_data.append(df_sorted.iloc[start:idx])
            
    if pre_fail_data:
        df_pre_fail = pd.concat(pre_fail_data)
        print("\nMédias 1 a 3 ciclos ANTES da falha (Anomalia Preliminar):")
        print(df_pre_fail[vars_to_analyze].mean())
        
        print("\n=== Correlação com a Falha ===")
        print(df_sorted[vars_to_analyze + ['Fail']].corr()['Fail'].sort_values(ascending=False))

if __name__ == "__main__":
    filepath = 'Test-O_G_Equipment_Data.csv'
    if os.path.exists(filepath):
        print("Iniciando Análise de Causa Raiz (Root Cause Analysis)...")
        df = load_data(filepath)
        vars_to_analyze = ['Temperature', 'Pressure', 'VibrationX', 'VibrationY', 'VibrationZ', 'Frequency']
        
        print("1. Gerando Violin Plots...")
        plot_distributions(df, vars_to_analyze)
        
        print("2. Gerando Pairplots...")
        plot_pairplots(df, vars_to_analyze)
        
        print("3. Gerando Análise por Presets...")
        plot_preset_comparison(df, vars_to_analyze)
        
        print("4. Detectando Padrões e Anomalias Pré-Falha...")
        feature_engineering_anomalies(df, vars_to_analyze)
        
        print("\nAnálise concluída. Imagens salvas como arquivos .png.")
    else:
        print("Arquivo de dados não encontrado.")
