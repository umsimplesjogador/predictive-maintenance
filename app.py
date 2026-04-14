from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Predictive Maintenance API", version="1.0")

# Load model path
MODEL_PATH = "xgboost_model.pkl"
model = None

# Define input data schema
class SensorData(BaseModel):
    Preset_1: int
    Preset_2: int
    Temperature: float
    Pressure: float
    VibrationX: float
    VibrationY: float
    VibrationZ: float
    Frequency: float
    # rolling features from feature_engineering
    Temperature_roll_mean_3: float
    Temperature_roll_std_3: float
    Temperature_diff: float
    Pressure_roll_mean_3: float
    Pressure_roll_std_3: float
    Pressure_diff: float
    VibrationX_roll_mean_3: float
    VibrationX_roll_std_3: float
    VibrationX_diff: float
    VibrationY_roll_mean_3: float
    VibrationY_roll_std_3: float
    VibrationY_diff: float
    VibrationZ_roll_mean_3: float
    VibrationZ_roll_std_3: float
    VibrationZ_diff: float
    Frequency_roll_mean_3: float
    Frequency_roll_std_3: float
    Frequency_diff: float

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

@app.get("/")
def home():
    return {"message": "Predictive Maintenance Equipment Failure API"}

@app.post("/predict")
def predict_failure(data: SensorData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    # Convert input data to DataFrame for prediction
    df = pd.DataFrame([data.dict()])
    
    # Predict probability of failure
    prob = model.predict_proba(df)[0][1]
    prediction = int(prob > 0.5)
    
    return {
        "failure_probability": float(prob),
        "prediction": prediction,
        "status": "Warning: Probable Failure" if prediction == 1 else "Normal Operation"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
