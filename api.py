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
# ML Model
# -----------------------------
yield_model = load("models/yield_model.joblib")

# -----------------------------
# Earth Engine Init (Render-safe)
# -----------------------------
import json

service_account = os.getenv("GEE_SERVICE_ACCOUNT")
key_json = os.getenv("GEE_KEY_JSON")

if not service_account or not key_json:
    raise Exception("GEE credentials not configured")

credentials = ee.ServiceAccountCredentials(
    service_account,
    key_data=json.loads(key_json),
)

ee.Initialize(credentials)
print("✅ Earth Engine initialized")

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

class NDVIRequest(BaseModel):
    lat: float
    lon: float
    boundary: list | None = None

# -----------------------------
# Helpers
# -----------------------------
def add_ndvi(image):
    return image.addBands(image.normalizedDifference(["B8", "B4"]).rename("NDVI"))

# -----------------------------
# Routes
# -----------------------------


# -----------------------------
# Yield Prediction
# -----------------------------
@app.post("/predict-yield")
def predict_yield(data: YieldInput):
    user_id = data.user_id or "guest_user"

    if not can_use(user_id):
        raise HTTPException(status_code=402, detail="Free limit exceeded")

    df = pd.DataFrame([data.dict(exclude={"user_id", "field_id"})])
    y = float(yield_model.predict(df)[0])

    confidence = 70
    if data.ndvi > 0.6:
        confidence += 10
    if data.humidity > 60:
        confidence += 5
    confidence = min(confidence, 95)

    add_yield_record(user_id=user_id, field_id=data.field_id, predicted=y)

    return {
        "predicted_yield": round(y, 2),
        "confidence": confidence,
        "history": get_history(user_id, data.field_id) or [
            {"year": 2021, "yield": 2.1},
            {"year": 2022, "yield": 2.6},
            {"year": 2023, "yield": 2.9}
        ],
        "ndvi_trend": [
            {"date": "2024-06-01", "ndvi": 0.42},
            {"date": "2024-07-01", "ndvi": 0.51},
            {"date": "2024-08-01", "ndvi": 0.58}
        ],
        "plan": get_user_plan(user_id)
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

