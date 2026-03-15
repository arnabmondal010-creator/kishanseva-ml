# api.py
# -*- coding: utf-8 -*-

import os, json, ee, pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load

from limits import can_use, get_user_plan, set_user_plan, mark_used
from ai_service import analyze_image
from services.yield_history_service import add_yield_record, get_history

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="KishanSeva AI API")
@app.get("/")
def root():
    return {"status": "KishanSeva AI running"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Models
# -----------------------------
class YieldInput(BaseModel):
    soil_type: str = "loamy"
    fertilizer_type: str = "urea"
    crop_stage: str = "vegetative"
    stress_level: str = "low"
    fertilizer_kg: float = 40
    irrigation_count: int = 2
    pesticide_sprays: int = 1
    avg_temp: float = 28
    rainfall: float = 0
    humidity: float = 60
    wind_speed: float = 1.5
    ndvi: float = 0.5
    user_id: str | None = "guest_user"
    field_id: str = "default"


# -----------------------------
# Yield Prediction
# -----------------------------

@app.post("/predict-yield")
def predict_yield(data: YieldInput):

    user_id = data.user_id or "guest_user"

    df = pd.DataFrame([{
        "soil_type": data.soil_type,
        "fertilizer_type": data.fertilizer_type,
        "crop_stage": data.crop_stage,
        "stress_level": data.stress_level,
        "fertilizer_kg": data.fertilizer_kg,
        "irrigation_count": data.irrigation_count,
        "pesticide_sprays": data.pesticide_sprays,
        "avg_temp": data.avg_temp,
        "rainfall": data.rainfall,
        "humidity": data.humidity,
        "wind_speed": data.wind_speed,
        "ndvi": data.ndvi
    }])

    y = float(yield_model.predict(df)[0])

    confidence = 70

    if data.ndvi > 0.6:
        confidence += 10

    if data.humidity > 60:
        confidence += 5

    confidence = min(confidence, 95)

    add_yield_record(user_id, data.field_id, y)

    return {
        "predicted_yield": round(y, 2),
        "confidence": confidence,
        "history": get_history(user_id, data.field_id)
    }

# -----------------------------
# NDVI (Live)
# -----------------------------
# api.py (ADD BELOW YOUR /ndvi-history OR ABOVE)

@app.post("/ndvi")
def field_ndvi(req: NDVIRequest):
    try:
        boundary = req.boundary

        if isinstance(boundary, str):
            boundary = json.loads(boundary)

        if boundary and isinstance(boundary, list) and len(boundary) > 2:
            coords = [[p["lon"], p["lat"]] for p in boundary if "lat" in p and "lon" in p]
            geom = ee.Geometry.Polygon([coords])
        else:
            geom = ee.Geometry.Point([req.lon, req.lat]).buffer(100)

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geom)
            .filterDate("2024-01-01", "2025-12-31")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .map(add_ndvi)
        )

        if collection.size().getInfo() == 0:
            return {"ndvi_mean": None, "status": "No data", "source": "Sentinel-2"}

        ndvi_img = collection.select("NDVI").mean()

        stats = ndvi_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=10,
            maxPixels=1e9,
        )

        ndvi_value = stats.get("NDVI").getInfo()

        if ndvi_value is None:
            return {"ndvi_mean": None, "status": "No NDVI", "source": "Sentinel-2"}

        return {
            "ndvi_mean": round(float(ndvi_value), 3),
            "status": "OK",
            "source": "Sentinel-2 (GEE)",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# -----------------------------
# NDVI Time-Series
# -----------------------------
@app.post("/ndvi-history")
def ndvi_history(req: NDVIRequest):
    try:
        geom = ee.Geometry.Point([req.lon, req.lat]).buffer(100)

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geom)
            .filterDate("2024-01-01", "2026-12-31")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            .map(add_ndvi)
            .select("NDVI")
        )

        if collection.size().getInfo() == 0:
            return {"ndvi_trend": []}

        def to_feature(img):
            mean = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=10,
                maxPixels=1e9,
            ).get("NDVI")

            return ee.Feature(None, {
                "date": ee.Date(img.get("system:time_start")).format("YYYY-MM-dd"),
                "ndvi": mean,
            })

        fc = collection.map(to_feature).filter(ee.Filter.notNull(["ndvi"]))
        data = fc.aggregate_array("date").zip(fc.aggregate_array("ndvi")).getInfo()

        history = [{"date": d, "ndvi": round(float(v), 3)} for d, v in data]

        return {"ndvi_trend": history}

    except Exception as e:
        return {"ndvi_trend": [], "error": str(e)}

from fastapi import Response

@app.head("/")
def root_head():
    return Response(status_code=200)

# -----------------------------
# Irrigation ML Prediction
# -----------------------------
import joblib
import pandas as pd
from pydantic import BaseModel

# Load ML model
irrigation_model = joblib.load("irrigation_model.pkl")


class IrrigationInput(BaseModel):
    soil: str
    crop: str
    temperature: float
    humidity: float
    rainfall: float
    ndvi: float
    infiltration: float


@app.post("/predict-irrigation")
def predict_irrigation(data: IrrigationInput):

    df = pd.DataFrame([data.dict()])

    prediction = irrigation_model.predict(df)[0]

    return {
        "irrigation_mm": round(float(prediction), 2)
    }

