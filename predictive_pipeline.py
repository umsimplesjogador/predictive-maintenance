import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score, roc_curve, classification_report, auc
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess(filepath):
    # Load data
    df = pd.read_csv(filepath)
    
    # Fill empty Fail values with 0
    df['Fail'] = df['Fail'].fillna(0).astype(int)
    
    # Ensure it's sorted by Cycle
    df = df.sort_values('Cycle').reset_index(drop=True)
    return df

def feature_engineering(df):
    df_feat = df.copy()
    
    # Sensor columns
    sensor_cols = ['Temperature', 'Pressure', 'VibrationX', 'VibrationY', 'VibrationZ', 'Frequency']
    
    # Create rolling features (window size 3 and 5 cycles)
    for col in sensor_cols:
        df_feat[f'{col}_roll_mean_3'] = df_feat[col].rolling(window=3, min_periods=1).mean()
        df_feat[f'{col}_roll_std_3'] = df_feat[col].rolling(window=3, min_periods=1).std().fillna(0)
        df_feat[f'{col}_diff'] = df_feat[col].diff().fillna(0)
        
    # Drop identifying or non-predictive columns from features
    # Keeping Preset_1 and Preset_2 as numerical categories or features
    df_feat = df_feat.drop(['Cycle'], axis=1)
    
    return df_feat

def train_and_evaluate(df):
    # Define features and target
    X = df.drop(['Fail'], axis=1)
    y = df['Fail']
    
    # Time-based split (80-20) to avoid data leakage (no cross-validation with future data)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train size: {len(X_train)} (Failures: {y_train.sum()})")
    print(f"Test size: {len(X_test)} (Failures: {y_test.sum()})")
    
    # Model 1: Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    rf_preds = rf_model.predict(X_test)
    
    # Model 2: XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=4, scale_pos_weight=(len(y_train)-y_train.sum())/y_train.sum(), random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_preds = xgb_model.predict(X_test)
    
    # Evaluation function
    def eval_model(name, y_true, y_pred, y_probs):
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_probs)
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(recall, precision)
        
        print(f"\n--- {name} ---")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC:  {roc_auc:.4f}")
        print(f"PR-AUC:   {pr_auc:.4f}")
        print(classification_report(y_true, y_pred))
        return {'f1': f1, 'roc_auc': roc_auc, 'pr_auc': pr_auc, 'precision': precision, 'recall': recall}
        
    res_rf = eval_model("Random Forest", y_test, rf_preds, rf_probs)
    res_xgb = eval_model("XGBoost", y_test, xgb_preds, xgb_probs)
    
    # Plot PR Curve
    plt.figure(figsize=(8, 6))
    plt.plot(res_rf['recall'], res_rf['precision'], label=f"Random Forest (AUC = {res_rf['pr_auc']:.2f})")
    plt.plot(res_xgb['recall'], res_xgb['precision'], label=f"XGBoost (AUC = {res_xgb['pr_auc']:.2f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig('pr_curve.png')
    plt.close()
    
    return xgb_model, X_train, X_test

def explain_model(model, X_train, X_test):
    print("\nCalculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.close()
    
    print("SHAP explicit feature importance and summary plot saved.")

if __name__ == "__main__":
    filepath = "Test-O_G_Equipment_Data.csv"
    df = load_and_preprocess(filepath)
    df_feat = feature_engineering(df)
    
    best_model, X_train, X_test = train_and_evaluate(df_feat)
    
    # Using XGBoost for Explainability as it usually has strong performance
    explain_model(best_model, X_train, X_test)
    print("Pipeline executado com sucesso.")
