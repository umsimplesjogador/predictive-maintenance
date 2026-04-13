import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Carrega o dataset e realiza as etapas de pré-processamento,
    incluindo tratamento de nulos e tipagem apropriada.
    
    Args:
        file_path (str): Caminho para o arquivo CSV de entrada.
        
    Returns:
        pd.DataFrame: DataFrame tratado e pronto para análise.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
        raise
    
    # Conferindo dados nulos e realizando casting / tratamento 
    # A coluna Fail possui nulos onde não houve falha, e "1" onde falhou.
    df['Fail'] = df['Fail'].fillna(0).astype(int)
    
    # Assegurando tipagens (Cycle e Presets como inteiros, variáveis contínuas como float)
    int_cols = ['Cycle', 'Preset_1', 'Preset_2']
    num_cols = ['Temperature', 'Pressure', 'VibrationX', 'VibrationY', 'VibrationZ', 'Frequency']
    
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
            
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
            
    return df


def plot_failure_distribution(df: pd.DataFrame) -> None:
    """
    Tarefa 1: Calcula e plota a quantidade de vezes que o equipamento falhou.
    
    Args:
        df (pd.DataFrame): DataFrame pré-processado.
    """
    fail_counts = df['Fail'].value_counts()
    fail_rate = (fail_counts.get(1, 0) / len(df)) * 100
    
    print("-" * 50)
    print("TAREFA 1: Distribuição de Falhas")
    print(f"Total de Registros: {len(df)}")
    print(f"Total de Falhas: {fail_counts.get(1, 0)}")
    print(f"Taxa de Falha: {fail_rate:.2f}%")
    print("-" * 50)
    
    # Visualização
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x='Fail', palette='viridis', hue='Fail', legend=False)
    plt.title('Distribuição de Falhas no Equipamento (0 = Normal, 1 = Falha)', fontsize=14)
    plt.xlabel('Status de Operação (Fail)', fontsize=12)
    plt.ylabel('Frequência (Quantidade)', fontsize=12)
    
    # Adicionando rótulos nas barras
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), 
                    textcoords='offset points')
        
    plt.tight_layout()
    plt.savefig('task1_failure_distribution.png', dpi=300)
    plt.show()


def plot_bivariate_presets(df: pd.DataFrame) -> None:
    """
    Tarefa 2: Categoriza e cria visualizações mostrando a distribuição 
    de falhas pelas configurações de Preset_1 e Preset_2. 
    Analisa se existe alguma combinação que sobrecarrega a máquina.
    
    Args:
        df (pd.DataFrame): DataFrame pré-processado contendo 'Preset_1', 'Preset_2' e 'Fail'.
    """
    print("-" * 50)
    print("TAREFA 2: Falhas por Configurações (Preset_1 e Preset_2)")
    
    # Resumo das taxas de falha por combinação de Preset_1 e Preset_2
    comb_fail = df.groupby(['Preset_1', 'Preset_2'])['Fail'].agg(['count', 'sum'])
    comb_fail['fail_rate'] = comb_fail['sum'] / comb_fail['count']
    comb_fail = comb_fail.sort_values('fail_rate', ascending=False)
    
    print("\nTop 5 Combinações com maior risco de falha (Sobrecarga):")
    print(comb_fail.head(5))
    print("-" * 50)
    
    # Visualizações de Preset Individual
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.barplot(data=df, x='Preset_1', y='Fail', ax=axes[0], palette='Blues_d', ci=None)
    axes[0].set_title('Taxa de Falha por Preset 1')
    axes[0].set_ylabel('Taxa Média de Falha')
    
    sns.barplot(data=df, x='Preset_2', y='Fail', ax=axes[1], palette='Oranges_d', ci=None)
    axes[1].set_title('Taxa de Falha por Preset 2')
    axes[1].set_ylabel('Taxa Média de Falha')
    
    plt.tight_layout()
    plt.savefig('task2_preset_individual.png', dpi=300)
    plt.show()

    # Visualização de Mapa de Calor (Combinação)
    plt.figure(figsize=(10, 6))
    pivot_fail = df.pivot_table(index='Preset_1', columns='Preset_2', values='Fail', aggfunc='mean')
    sns.heatmap(pivot_fail, annot=True, cmap='RdYlGn_r', fmt='.2%', linewidths=.5)
    plt.title('Heatmap: Taxa Média de Falhas por Combinação de Presets', fontsize=14)
    plt.xlabel('Preset 2', fontsize=12)
    plt.ylabel('Preset 1', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('task2_preset_heatmap.png', dpi=300)
    plt.show()

    
def main():
    file_path = 'Test-O_G_Equipment_Data.csv'
    
    try:
        # Pre-processamento
        df = load_and_preprocess_data(file_path)
        
        # Análises
        plot_failure_distribution(df)
        plot_bivariate_presets(df)
        
    except Exception as e:
        print(f"Erro durante a execução do processo: {e}")

if __name__ == "__main__":
    main()
