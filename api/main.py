# =============================================================================
# PREDICTIVE MAINTENANCE — FASTAPI
# File    : api/main.py
# Run     : uvicorn api.main:app --reload
# Install : pip install fastapi uvicorn joblib scikit-learn pandas numpy
# =============================================================================

import numpy as np
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =============================================================================
# Load model, scaler, threshold
# (these files must be in the root project folder)
# =============================================================================
model     = joblib.load('model.pkl')
scaler    = joblib.load('scaler.pkl')
threshold = joblib.load('threshold.pkl')

# =============================================================================
# App setup
# =============================================================================
app = FastAPI(
    title       = "Predictive Maintenance API",
    description = "Predicts machine failure from sensor readings using Random Forest",
    version     = "1.0.0"
)

# Allow Streamlit dashboard to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# =============================================================================
# Input schema
# Pydantic validates every field automatically — wrong type = clear error message
# =============================================================================
class SensorInput(BaseModel):
    Type                      : int   = Field(..., ge=0, le=2,    description="Machine type: L=0, M=1, H=2")
    air_temperature           : float = Field(..., ge=295, le=305, description="Air temperature in Kelvin")
    process_temperature       : float = Field(..., ge=305, le=315, description="Process temperature in Kelvin")
    rotational_speed          : int   = Field(..., ge=1000, le=3000, description="Rotational speed in RPM")
    torque                    : float = Field(..., ge=0, le=80,    description="Torque in Nm")
    tool_wear                 : int   = Field(..., ge=0, le=300,   description="Tool wear in minutes")

# =============================================================================
# Helper: build full feature vector (raw + engineered)
# Must match EXACTLY what Phase 3 produced
# =============================================================================
def build_features(data: SensorInput) -> np.ndarray:
    power     = data.torque * (data.rotational_speed * 2 * 3.14159 / 60)
    temp_diff = data.process_temperature - data.air_temperature
    strain    = data.tool_wear * data.torque

    features = [[
        data.Type,
        data.air_temperature,
        data.process_temperature,
        data.rotational_speed,
        data.torque,
        data.tool_wear,
        power,
        temp_diff,
        strain,
    ]]
    return features

# =============================================================================
# Routes
# =============================================================================

@app.get("/")
def root():
    return {
        "message"   : "Predictive Maintenance API is running",
        "docs"      : "/docs",
        "predict"   : "/predict"
    }

@app.get("/health")
def health():
    return {"status": "ok", "threshold": threshold}

@app.post("/predict")
def predict(data: SensorInput):
    # 1. Build feature vector
    features = build_features(data)

    # 2. Scale using same scaler from Phase 3
    features_scaled = scaler.transform(features)

    # 3. Get failure probability
    probability = model.predict_proba(features_scaled)[0][1]

    # 4. Apply tuned threshold (0.65) not default (0.50)
    prediction = int(probability >= threshold)

    # 5. Return result
    return {
        "prediction"        : prediction,
        "result"            : "⚠️ FAILURE PREDICTED" if prediction == 1 else "✅ NORMAL OPERATION",
        "failure_probability": round(float(probability), 4),
        "threshold_used"    : threshold,
        "input_received"    : data.dict(),
    }