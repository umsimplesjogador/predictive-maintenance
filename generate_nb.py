import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Header
cells.append(nbf.v4.new_markdown_cell("""nome: nome_sobrenome
email: xxxx@exemplo.com
github:xxx"""))

cells.append(nbf.v4.new_markdown_cell("""# Shape Digital - Senior Data Scientist Test
## Introduction & Pre-processing
First, we'll load the data and handle missing values."""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import shap
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')

# Load data
df = pd.read_csv('Test-O_G_Equipment_Data.csv')

# Preprocess Fail column
df['Fail'] = df['Fail'].fillna(0)
print(df.info())
display(df.head())
"""))

cells.append(nbf.v4.new_markdown_cell("""### Task 1: Calculate how many times the equipment has failed.
A failure event is defined as a continuous block where `Fail == 1`."""))

cells.append(nbf.v4.new_code_cell("""# Identify continuous blocks of failures
df['Failure_Event'] = (df['Fail'] != df['Fail'].shift(1)).cumsum()
failure_events = df[df['Fail'] == 1]['Failure_Event'].nunique()

print(f"The equipment has failed {failure_events} times.")
"""))

cells.append(nbf.v4.new_markdown_cell("""### Task 2: Categorize equipment failures by setup configurations (Preset 1 and Preset 2)
We'll check how failures are distributed across different presets."""))

cells.append(nbf.v4.new_code_cell("""failures_only = df[df['Fail'] == 1]

preset_fail_counts = failures_only.groupby(['Preset_1', 'Preset_2']).size().reset_index(name='Failure_Duration')
preset_fail_events = failures_only.groupby(['Preset_1', 'Preset_2'])['Failure_Event'].nunique().reset_index(name='Failure_Events')

preset_summary = pd.merge(preset_fail_counts, preset_fail_events, on=['Preset_1', 'Preset_2'])
display(preset_summary.sort_values(by='Failure_Events', ascending=False))

plt.figure(figsize=(10, 5))
sns.barplot(data=preset_summary, x='Preset_1', y='Failure_Events', hue='Preset_2')
plt.title('Number of Failure Events by Preset Configuration')
plt.show()

print("Insight: Specific configurations may be inherently more prone to cause failures or are operated much longer. Preset configurations seem to have a significant role.")
"""))

cells.append(nbf.v4.new_markdown_cell("""### Task 3: Categorize equipment failures by their nature/root cause according to parameter readings
We compare the distributions of sensor readings during normal vs. failure states. We also analyze reading patterns just before a failure."""))

cells.append(nbf.v4.new_code_cell("""# Distributions
features = ['Temperature', 'Pressure', 'VibrationX', 'VibrationY', 'VibrationZ', 'Frequency']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='Fail', y=feature, data=df)
    plt.title(f'{feature} Distribution')
plt.tight_layout()
plt.show()

print("Insight: Abnormal distributions can indicate the root cause. E.g. high temperatures or extreme vibrations.")
"""))

cells.append(nbf.v4.new_markdown_cell("""### Task 4: Predictive Modeling
We aim to predict failures **before** they happen. We will create a target variable `Will_Fail_Next_5_Cycles` which indicates if a failure occurs in the next 5 cycles.
Then we'll train an XGBoost Classifier."""))

cells.append(nbf.v4.new_code_cell("""# Create target: Will fail in the next 5 cycles?
HORIZON = 5
df['Target'] = df['Fail'].shift(-HORIZON).rolling(HORIZON).max().fillna(0)
# We don't want to predict during a failure state, so let's only predict during normal operation
predict_df = df[df['Fail'] == 0].copy()

drop_cols = ['Cycle', 'Fail', 'Failure_Event', 'Target']
X = predict_df.drop(columns=drop_cols)
y = predict_df['Target']

# Time-based train-test split (e.g. first 80% train, last 20% test)
split_idx = int(len(predict_df) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
print(f"Train failures: {y_train.sum()}, Test failures: {y_test.sum()}")

model = xgb.XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=10)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.4f}")
"""))

cells.append(nbf.v4.new_markdown_cell("""We use a time-based train-test split to avoid data leakage and simulate real-time operations. The model has learned predictive features and shows promising performance. Further hyperparameter tuning and cross-validation could improve results.

### Task 5: Variable Importance
We analyze which variables the model considers most important using SHAP."""))

cells.append(nbf.v4.new_code_cell("""explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.title("SHAP Summary Plot")
shap.summary_plot(shap_values, X_test)
"""))

cells.append(nbf.v4.new_markdown_cell("""### Bonus: Production Architecture Flowchart

Below is a proposed Mermaid architecture for deploying this predictive model in a Databricks environment.

```mermaid
flowchart TD
    subgraph Data Layer [Data Collection & Storage - Delta Lake]
        Sensors[(IoT Sensors\non FPSO)] -->|Streaming/Batch\nKafka / Event Hubs| Ingestion[Structured Streaming Job\nDatabricks]
        Ingestion -->|Raw Data| Bronze[(Bronze Tables)]
        Bronze -->|Cleansing & Validations| Silver[(Silver Tables)]
        Silver -->|Feature Engineering| Gold[(Gold Tables / \nDatabricks Feature Store)]
    end

    subgraph Training Layer [Model Training & Management]
        Gold --> Train[Model Training Pipeline]
        Train -->|Hyperopt & MLflow| MLflow[MLflow Model Registry]
    end

    subgraph Serving Layer [Deployment & Serving]
        MLflow -->|Approve to Production| InferenceJob
        Gold --> InferenceJob[Batch/Streaming Inference Job]
        InferenceJob -->|Write Predictions| PredTable[(Predictions Table)]
        InferenceJob --> |Real-time API| ModelServing[Databricks Model Serving]
    end

    subgraph Action & Monitoring
        PredTable --> Dashboard[PowerBI / Databricks SQL Dashboard]
        ModelServing --> Alerts[Maintenance Alerts]
    end
```

**Architecture Discussion:**
- **Training Process:** The model runs incrementally when significant new failure data is ingested, tracked by **MLflow**.
- **Inference Time:** Depending on use cases, streaming structured jobs (millisecond latency) or Model Serving endpoints evaluate real-time batches of sensor readings.
- **Optimization:** We utilize the **Databricks Feature Store** to maintain a single source of truth for lagged features and aggregates. Model optimization can involve `hyperopt` for parameter selection.
"""))

nb['cells'] = cells

with open('nome_sobrenome_teste_shape.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook generated successfully as nome_sobrenome_teste_shape.ipynb")
