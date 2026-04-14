import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, precision_recall_curve, auc
import warnings
import optuna
import mlflow
import json

warnings.filterwarnings('ignore')

def load_and_preprocess(filepath):
    print(">>> Loading and Preprocessing Data...")
    df = pd.read_csv(filepath)
    df = df.sort_values(by='Cycle').reset_index(drop=True)
    df['Fail'] = df['Fail'].fillna(0).astype(int)
    
    sensor_cols = ['Temperature', 'Pressure', 'VibrationX', 'VibrationY', 'VibrationZ', 'Frequency']
    for col in sensor_cols:
        df[col] = df[col].ffill().bfill()
        
    return df

def feature_engineering(df):
    print("\n>>> Performing Feature Engineering...")
    HORIZON = 5
    df['Target'] = df['Fail'].shift(-HORIZON).rolling(HORIZON).max().fillna(0)
    
    predict_df = df[df['Fail'] == 0].copy()
    
    sensor_cols = ['Temperature', 'Pressure', 'VibrationX', 'VibrationY', 'VibrationZ', 'Frequency']
    for col in sensor_cols:
        predict_df[f'{col}_roll_mean_3'] = predict_df[col].shift(1).rolling(window=3, min_periods=1).mean().fillna(predict_df[col])
        predict_df[f'{col}_roll_std_3'] = predict_df[col].shift(1).rolling(window=3, min_periods=1).std().fillna(0)
        predict_df[f'{col}_diff'] = predict_df[col].diff().fillna(0)
    
    drop_cols = ['Fail', 'Target']
    if 'Failure_Event' in predict_df.columns:
        drop_cols.append('Failure_Event')
        
    X = predict_df.drop(columns=drop_cols)
    y = predict_df['Target']
    
    return X, y

def time_based_split(X, y, test_ratio=0.2):
    print("\n>>> Temporal Splitting...")
    split_idx = int(len(X) * (1 - test_ratio))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    X_train = X_train.drop(columns=['Cycle'])
    X_test = X_test.drop(columns=['Cycle'])
    
    return X_train, X_test, y_train, y_test

def log_metrics(y_true, y_pred, y_probs, prefix=""):
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall_curve, precision_curve)
    
    metrics = {
        f"{prefix}f1_score": f1,
        f"{prefix}precision": prec,
        f"{prefix}recall": rec,
        f"{prefix}accuracy": acc,
        f"{prefix}roc_auc": roc_auc,
        f"{prefix}pr_auc": pr_auc
    }
    return metrics

def tune_and_train(X_train, X_test, y_train, y_test):
    print("\n>>> MLflow Optuna Orchestration for Multi-Models...")
    scale_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9)
    print(f"Base Imbalance weight (scale_pos_weight for trees): {scale_weight:.2f}")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Predictive_Maintenance_Models")

    def objective(trial):
        classifier_name = trial.suggest_categorical("classifier", ["RandomForest", "LightGBM", "XGBoost"])
        
        with mlflow.start_run(nested=True, run_name=classifier_name):
            if classifier_name == "RandomForest":
                params = {
                    'n_estimators': trial.suggest_int('rf_n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('rf_max_depth', 3, 10),
                    'class_weight': 'balanced',
                    'random_state': 42
                }
                model = RandomForestClassifier(**params)
                
            elif classifier_name == "LightGBM":
                params = {
                    'n_estimators': trial.suggest_int('lgb_n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('lgb_max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.2),
                    'scale_pos_weight': scale_weight,
                    'random_state': 42,
                    'verbose': -1
                }
                model = lgb.LGBMClassifier(**params)
                
            else: # XGBoost
                params = {
                    'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.2),
                    'scale_pos_weight': scale_weight,
                    'eval_metric': 'logloss',
                    'random_state': 42
                }
                model = xgb.XGBClassifier(**params)

            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_probs = model.predict_proba(X_test)[:, 1]
            
            metrics = log_metrics(y_test, y_pred, y_probs)
            
            mlflow.log_param("classifier", classifier_name)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            
            # Using PR_AUC as the main target for Optuna since data is highly imbalanced
            # Alternative: F1-Score
            return metrics["pr_auc"]

    study = optuna.create_study(direction="maximize")
    with mlflow.start_run(run_name="Optuna_HyperSearch"):
        # Limited to 20 trials for demonstration and time limits
        study.optimize(objective, n_trials=20)
        
        print("\n=== Best Trial ===")
        trial = study.best_trial
        print(f"  PR-AUC Value: {trial.value}")
        print("  Params: ")
        
        # We need to extract the best params to retrain the final model
        best_params = {}
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
            mlflow.log_param(f"best_{key}", value)
            if key != "classifier":
                # Remove prefix from param names to instantiate model
                clean_key = "_".join(key.split("_")[1:])
                best_params[clean_key] = value

        classifier_name = trial.params["classifier"]
        
        print(f"\n>>> Retraining Best Model: {classifier_name}")
        best_params['random_state'] = 42
        
        if classifier_name == "RandomForest":
            best_params['class_weight'] = 'balanced'
            final_model = RandomForestClassifier(**best_params)
        elif classifier_name == "LightGBM":
            best_params['scale_pos_weight'] = scale_weight
            best_params['verbose'] = -1
            final_model = lgb.LGBMClassifier(**best_params)
        else:
            best_params['scale_pos_weight'] = scale_weight
            best_params['eval_metric'] = 'logloss'
            final_model = xgb.XGBClassifier(**best_params)
            
        final_model.fit(X_train, y_train)
        
        y_pred = final_model.predict(X_test)
        y_probs = final_model.predict_proba(X_test)[:, 1]
        
        final_metrics = log_metrics(y_test, y_pred, y_probs)
        
        print("\n>>> Final Evaluation metrics:")
        for k, v in final_metrics.items():
            print(f"{k}: {v:.4f}")
            
        with open('final_metrics.json', 'w') as f:
            final_metrics['best_model'] = classifier_name
            for k, v in trial.params.items():
                if k != "classifier":
                   final_metrics[f"hiperparam_{k}"] = v
            json.dump(final_metrics, f, indent=4)
        
        mlflow.sklearn.log_model(final_model, "best_model_artifact")
        
        return final_model, final_metrics

if __name__ == "__main__":
    filepath = "Test-O_G_Equipment_Data.csv"
    df = load_and_preprocess(filepath)
    X, y = feature_engineering(df)
    
    X_train, X_test, y_train, y_test = time_based_split(X, y)
    
    best_model, metrics = tune_and_train(X_train, X_test, y_train, y_test)
    
    print("\n>>> Exporting local independent pickle for API deployment...")
    joblib.dump(best_model, 'xgboost_model.pkl')  
    # Still naming it xgboost_model.pkl globally to prevent breaking app.py changes
    print("Model successfully exported as 'xgboost_model.pkl'.")
