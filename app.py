from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI(title="KishanSeva Yield ML API")

# ---------------- INPUT SCHEMA ----------------
class PredictRequest(BaseModel):
    ndvi: float
    rainfall: float
    temperature: float
    fertilizerKg: float
    irrigationCount: int
    pesticideCount: int
    cropStage: str


# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("❌ model.pkl not found. Run train_model.py first.")

model = joblib.load(MODEL_PATH)
print("✅ Model loaded with cropStage")


# ---------------- HEALTH CHECK ----------------
@app.get("/")
def health():
    return {
        "service": "KishanSeva Yield Prediction",
        "status": "running"
    }


# ---------------- PREDICTION ----------------
@app.post("/predict-yield")
def predict_yield(data: PredictRequest):

    stage_map = {
        "sowing": 0,
        "vegetative": 1,
        "flowering": 2,
        "harvest": 3,
    }

    crop_stage_val = stage_map.get(data.cropStage.lower(), 1)

    features = np.array([[
        data.ndvi,
        data.rainfall,
        data.temperature,
        data.fertilizerKg,
        data.irrigationCount,
        data.pesticideCount,
        crop_stage_val
    ]])

    prediction = float(model.predict(features)[0])

    return {
        "estimatedYield": round(prediction, 2),
        "confidence": 0.82,
        "explanation": "RandomForest model using NDVI, weather, inputs, and crop stage"
    }
