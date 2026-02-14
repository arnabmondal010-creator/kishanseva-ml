import joblib
import pandas as pd

# Load model once on startup
model = joblib.load("models/yield_model.joblib")

def predict_yield(data: dict):
    """
    data = {
      soil_type, fertilizer_type, crop_stage, stress_level,
      fertilizer_kg, irrigation_count, pesticide_sprays,
      avg_temp, rainfall, humidity, wind_speed, ndvi
    }
    """

    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    return {
        "predicted_yield": round(float(prediction), 2),  # tons/ha
        "unit": "tons per hectare"
    }
