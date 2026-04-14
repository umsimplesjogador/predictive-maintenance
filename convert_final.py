import json

with open("notebooks/01_eda_storytelling.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Lendo o source do modelo
with open("src/data_preprocessing_and_modeling.py", "r", encoding="utf-8") as f:
    modelo_py = f.read()

celulas_modelo = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## O Motor Estrutural de Classificação O&G (Deploy Source)\n",
            "Abaixo, em anexo para avaliação integral, encontra-se a transcrição pura do código-fonte do *Pipeline Auto-Tuning de MLOps* rodado no servidor para forjar o Picke Model, integrando Random-Forests, LightGBM e XGBoost competindo simultaneamente contra otimizadores Bayesianos do **Optuna** na busca pelo Limiar PR-AUC e mitigação da penalidade classe `scale_pos_weight` descompensada."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "```python\n" + modelo_py + "\n```"
        ]
    }
]

nb["cells"].extend(celulas_modelo)

with open("notebooks/01_eda_storytelling.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

import subprocess
import os

# Executa Conversao Estática
print("Convertendo Master HTML via JUPYTER...")
subprocess.run(["python", "-m", "jupyter", "nbconvert", "--to", "html", "notebooks/01_eda_storytelling.ipynb", "--output", "../fabio_geliustique_da_silva_teste_shape.html"])
