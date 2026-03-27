# api.py
# -*- coding: utf-8 -*-

import os
import json
import ee
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from limits import can_use, get_user_plan, set_user_plan, mark_used
from ai_service import analyze_image
from services.yield_history_service import add_yield_record, get_history
from functools import lru_cache
import time

NDVI_CACHE = {}
CACHE_TTL = 3600  # 1 hour

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
# Load ML Models
# -----------------------------
yield_model = joblib.load("models/yield_model.joblib")
irrigation_model = joblib.load("irrigation_model.pkl")

# -----------------------------
# Earth Engine Init
# -----------------------------

service_account = os.getenv("GEE_SERVICE_ACCOUNT")
key_json = os.getenv("GEE_KEY_JSON")

if not service_account or not key_json:
    raise Exception("GEE credentials not configured")

credentials = ee.ServiceAccountCredentials(
    service_account,
    key_data=key_json,
)

ee.Initialize(credentials)

print("Earth Engine initialized")

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


class IrrigationInput(BaseModel):
    soil: str
    crop: str
    temperature: float
    humidity: float
    rainfall: float
    ndvi: float
    infiltration: float


# -----------------------------
# Helper
# -----------------------------
def add_ndvi(image):
    return image.addBands(
        image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    )

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

@app.post("/satellite-analysis")
def satellite_analysis(req: NDVIRequest):

    try:
        import ee
        import json

        # 🔥 CACHE
        key = f"{req.lat}_{req.lon}"
        if key in NDVI_CACHE:
            data, ts = NDVI_CACHE[key]
            if time.time() - ts < CACHE_TTL:
                return data

        boundary = req.boundary

        if isinstance(boundary, str):
            boundary = json.loads(boundary)

        # ================= GEOMETRY =================
        if boundary and isinstance(boundary, list) and len(boundary) > 2:

            coords = [
                [float(p["lon"]), float(p["lat"])]
                for p in boundary
                if "lat" in p and "lon" in p
            ]

            geom = ee.Geometry.Polygon([coords])

        else:
            geom = ee.Geometry.Point([req.lon, req.lat]).buffer(50)

        # ================= COLLECTION =================
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geom)
            .filterDate("2024-01-01", "2026-12-31")  # ✅ same as before
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        )

        if collection.size().getInfo() == 0:
            return {
                "status": "No satellite data",
                "latest": None,
                "history": [],
                "trend": None
            }

        # ================= INDICES =================
        def add_indices(img):

            nir  = img.select("B8")
            red  = img.select("B4")
            swir = img.select("B11")

            ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
            ndwi = nir.subtract(swir).divide(nir.add(swir)).rename("NDWI")
            savi = nir.subtract(red).divide(nir.add(red).add(0.5)).multiply(1.5).rename("SAVI")

            return img.addBands([ndvi, ndwi, savi])

        # ================= LATEST IMAGE =================
        latest_img = add_indices(
            collection.sort("system:time_start", False).first()
        )

        # ================= MASK =================
        mask = ee.Image.constant(1).clip(geom)

        ndvi_img = latest_img.select("NDVI").updateMask(mask).clip(geom)
        ndwi_img = latest_img.select("NDWI").updateMask(mask).clip(geom)
        savi_img = latest_img.select("SAVI").updateMask(mask).clip(geom)

        # ================= 🔥 DYNAMIC STRETCH (SAME AS BEFORE) =================
        ndwi_stats = ndwi_img.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=geom,
            scale=10,
            maxPixels=1e9,
        ).getInfo()

        savi_stats = savi_img.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=geom,
            scale=10,
            maxPixels=1e9,
        ).getInfo()

        ndwi_min = float(ndwi_stats.get("NDWI_min", -0.2))
        ndwi_max = float(ndwi_stats.get("NDWI_max", 0.2))

        savi_min = float(savi_stats.get("SAVI_min", 0))
        savi_max = float(savi_stats.get("SAVI_max", 1))

        # ================= VIS =================
        ndvi_vis = {
            "min": 0,
            "max": 1,
            "palette": ["#8b0000", "#ffcc00", "#006400"]
        }

        ndwi_vis = {
            "min": ndwi_min,
            "max": ndwi_max,
            "palette": ["#7f3b08", "#f7f7f7", "#2b83ba"]
        }

        savi_vis = {
            "min": savi_min,
            "max": savi_max,
            "palette": ["#5e4fa2", "#fdae61", "#1a9850"]
        }

        def get_tile(image, vis):
            try:
                m = ee.Image(image).getMapId(vis)
                return m["tile_fetcher"].url_format
            except Exception as e:
                print("🔥 TILE ERROR:", e)
                return None

        tiles = {
            "ndvi": get_tile(ndvi_img, ndvi_vis),
            "ndwi": get_tile(ndwi_img, ndwi_vis),
            "savi": get_tile(savi_img, savi_vis),
        }

        # ================= STATS =================
        stats = latest_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=10,
            maxPixels=1e9,
        ).getInfo()

        latest = {
            "date": ee.Date(
                latest_img.get("system:time_start")
            ).format("YYYY-MM-dd").getInfo(),

            "ndvi": round(float(stats.get("NDVI", 0)), 3),
            "ndwi": round(float(stats.get("NDWI", 0)), 3),
            "savi": round(float(stats.get("SAVI", 0)), 3),
        }

        # ================= HISTORY =================
        def to_feature(img):

            img = add_indices(img)

            mean = img.select("NDVI").reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=10,
                maxPixels=1e9,
            ).get("NDVI")

            return ee.Feature(
                None,
                {
                    "date": ee.Date(
                        img.get("system:time_start")
                    ).format("YYYY-MM-dd"),
                    "ndvi": mean,
                },
            )

        fc = collection.map(to_feature).filter(
            ee.Filter.notNull(["ndvi"])
        )

        data = fc.aggregate_array("date").zip(
            fc.aggregate_array("ndvi")
        ).getInfo()

        history = [
            {"date": d, "ndvi": round(float(v), 3)}
            for d, v in data
        ] if data else []

        # ================= TREND =================
        trend = None

        if len(history) >= 2:

            start = history[0]["ndvi"]
            end = history[-1]["ndvi"]

            change = round(end - start, 3)
            change_percent = round((change / start) * 100, 2) if start != 0 else 0

            if change > 0.03:
                trend_type = "improving"
            elif change < -0.03:
                trend_type = "declining"
            else:
                trend_type = "stable"

            trend = {
                "start_ndvi": start,
                "current_ndvi": end,
                "change": change,
                "change_percent": change_percent,
                "trend": trend_type
            }

        # ================= RESPONSE =================
        result = {
            "status": "OK",
            "latest": latest,
            "history": history[-12:] if history else [],
            "trend": trend,
            "tiles": tiles,
            "source": "Sentinel-2 (Google Earth Engine)"
        }

        NDVI_CACHE[key] = (result, time.time())

        return result

    except Exception as e:
        print("🔥 ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
# -----------------------------
# Irrigation Prediction
# -----------------------------
@app.post("/predict-irrigation")
def predict_irrigation(data: IrrigationInput):

    df = pd.DataFrame([data.dict()])

    prediction = irrigation_model.predict(df)[0]

    return {
        "irrigation_mm": round(float(prediction), 2)
    }


# -----------------------------
# HEAD health check
# -----------------------------
@app.head("/")
def root_head():
    return Response(status_code=200)

# -----------------------------------
# CROP DISEASE
# -----------------------------------

from fastapi import UploadFile, File, Form
from PIL import Image
import numpy as np
import io

@app.post("/predict-disease")
async def predict_disease(
    crop: str = Form(...),
    user_id: str = Form(...),
    file: UploadFile = File(...)
):

    try:

        contents = await file.read()

        print("Uploaded image size:", len(contents))   # ✅ INSIDE try

        result = await analyze_image(contents, crop)

        return {
            "crop": crop,
            "disease": result.get("disease"),
            "confidence": result.get("confidence"),
            "advice": result.get("advice")
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

from fastapi import Query
from sqlalchemy import create_engine, text


# ================= DB =================
DB_URL = os.getenv("DATABASE_URL")

if DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql+psycopg2://", 1)

engine = create_engine(DB_URL)


# ================= CACHE =================
@lru_cache(maxsize=100)
def cached_query(crop, district, sort, limit, offset):

    query = """
    SELECT commodity, district, market, price, date
    FROM market_prices
    WHERE 1=1
    """

    params = {}

    if crop:
        query += " AND LOWER(commodity) LIKE LOWER(:crop)"
        params["crop"] = f"%{crop}%"

    if district:
        query += " AND LOWER(district) LIKE LOWER(:district)"
        params["district"] = f"%{district}%"

    # 🔥 SAFE SORT
    if sort == "date":
        query += " ORDER BY date DESC"
    else:
        query += " ORDER BY price DESC"

    query += " LIMIT :limit OFFSET :offset"

    params["limit"] = limit
    params["offset"] = offset

    with engine.connect() as conn:
        result = conn.execute(text(query), params)
        rows = [dict(r._mapping) for r in result]

    return rows


# ================= APIfrom fastapi import Query
from sqlalchemy import text

@app.get("/market-prices")
def get_prices(
    crop: str = Query(default=None),
    district: str = Query(default=None),
    sort: str = Query(default="price"),
    limit: int = Query(default=20),
    offset: int = Query(default=0),
):

    query = """
    SELECT commodity, district, market, price, date
    FROM market_prices
    WHERE 1=1
    """

    params = {
        "limit": limit,
        "offset": offset
    }

    if crop:
        query += " AND LOWER(commodity) LIKE LOWER(:crop)"
        params["crop"] = f"%{crop}%"

    if district:
        query += " AND LOWER(district) LIKE LOWER(:district)"
        params["district"] = f"%{district}%"

    # 🔥 SORT
    if sort == "date":
        query += " ORDER BY date DESC"
    else:
        query += " ORDER BY price DESC"

    # 🔥 PAGINATION (THIS WAS MISSING)
    query += " LIMIT :limit OFFSET :offset"

    with engine.connect() as conn:
        result = conn.execute(text(query), params)
        rows = [dict(r._mapping) for r in result]

    return rows
#----------------------------------
# ================= FIREBASE =================

import os
import json
import firebase_admin
from firebase_admin import credentials, messaging

# 🔥 LOAD KEY
firebase_key = json.loads(os.environ["FIREBASE_KEY"])
cred = credentials.Certificate(firebase_key)

# 🔥 SAFE INIT (IMPORTANT FIX)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)


# ================= SEND SINGLE =================

@app.post("/send-notification")
def send_notification(token: str, title: str, body: str):

    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body,
        ),
        token=token,
    )

    response = messaging.send(message)

    return {"success": True, "id": response}


# ================= FIRESTORE =================

from google.cloud import firestore
from google.oauth2 import service_account

credentials_fs = service_account.Credentials.from_service_account_info(firebase_key)

db = firestore.Client(
    credentials=credentials_fs,
    project=firebase_key["project_id"],
)


# ================= NOTIFY ALL =================

@app.post("/notify-all")
def notify_all():

    users = db.collection("farmers").stream()

    sent = 0

    for user in users:
        data = user.to_dict()
        token = data.get("fcm_token")

        if not token:
            continue

        try:
            message = messaging.Message(
                notification=messaging.Notification(
                    title="KishanSeva Alert 🌱",
                    body="Check your farm updates today",
                ),
                token=token,
            )

            messaging.send(message)
            sent += 1

        except Exception as e:
            print("❌ Failed:", e)

    return {"sent": sent}


# ================= TOPIC =================

@app.post("/notify-topic")
def notify_topic(title: str, body: str):

    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body,
        ),
        topic="all_users",
    )

    messaging.send(message)

    return {"success": True}


# ================= DAILY =================

@app.get("/daily-reminder")
def daily_reminder():

    message = messaging.Message(
        notification=messaging.Notification(
            title="Daily Reminder 🌱",
            body="Check your crop health today",
        ),
        topic="all_users",
    )

    messaging.send(message)

    return {"sent": True}


# ================= HELPERS =================

import requests

def get_weather(lat, lon):

    key = os.getenv("OPENWEATHER_API_KEY")

    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}&units=metric"

    try:
        res = requests.get(url, timeout=5).json()  # 🔥 timeout added
        return res.get("weather", [{}])[0].get("main", "").lower()
    except:
        return ""


def get_ndvi(lat, lon):

    url = "https://kishanseva-ai.onrender.com/satellite-analysis"

    try:
        res = requests.post(
            url,
            json={"lat": lat, "lon": lon},
            timeout=5  # 🔥 timeout added
        ).json()

        if res.get("latest"):
            return res["latest"]["ndvi"]

    except:
        return None

    return None


def get_users():

    users = db.collection("farmers").stream()

    data = []

    for u in users:
        d = u.to_dict()

        if d.get("lat") and d.get("lon"):

            data.append({
                "token": d.get("fcm_token"),
                "lat": d.get("lat"),
                "lon": d.get("lon"),
            })

    return data


# ================= SMART ALERT =================

from datetime import datetime

@app.get("/smart-alerts")
def smart_alerts():

    users = get_users()
    sent = 0

    for u in users:

        lat = u.get("lat")
        lon = u.get("lon")
        token = u.get("token")

        if not token or lat is None or lon is None:
            continue

        try:

            # ================= FETCH WEATHER =================
            key = os.getenv("OPENWEATHER_API_KEY")
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}&units=metric"

            weather_res = requests.get(url, timeout=5).json()

            weather = weather_res.get("weather", [{}])[0].get("main", "").lower()
            temp = weather_res.get("main", {}).get("temp")
            humidity = weather_res.get("main", {}).get("humidity")

            # ================= NDVI =================
            ndvi = get_ndvi(lat, lon)

            # ================= LOAD PREVIOUS =================
            user_ref = db.collection("alerts_state").document(token)
            prev = user_ref.get().to_dict() or {}

            prev_ndvi = prev.get("ndvi")

            alerts = []

            # =====================================================
            # 🌱 NDVI INTELLIGENCE
            # =====================================================

            if ndvi is not None:

                if prev_ndvi is not None:
                    change = ndvi - prev_ndvi

                    if change < -0.1:
                        alerts.append(("🚨 Crop Declining",
                                       f"NDVI dropped {prev_ndvi:.2f} → {ndvi:.2f}"))

                    elif change > 0.1:
                        alerts.append(("🌱 Crop Improving",
                                       f"NDVI improved {prev_ndvi:.2f} → {ndvi:.2f}"))

                if ndvi < 0.3:
                    alerts.append(("🚨 Critical Crop",
                                   f"Very low NDVI ({ndvi:.2f})"))

                elif ndvi < 0.5:
                    alerts.append(("⚠️ Moderate Crop",
                                   f"NDVI moderate ({ndvi:.2f})"))

                else:
                    alerts.append(("✅ Healthy Crop",
                                   f"NDVI good ({ndvi:.2f})"))

            # =====================================================
            # 🌦 WEATHER INTELLIGENCE
            # =====================================================

            if weather:

                if "rain" in weather:
                    alerts.append(("🌧 Rain Incoming",
                                   "Rain expected. Avoid irrigation"))

                elif "cloud" in weather:
                    alerts.append(("☁️ Cloudy",
                                   "Low sunlight may affect growth"))

                else:
                    alerts.append(("☀️ Clear Weather",
                                   "Good farming conditions"))

            if temp is not None:

                if temp > 35:
                    alerts.append(("🔥 High Temperature",
                                   f"{temp}°C → crop stress risk"))

                elif temp < 15:
                    alerts.append(("❄️ Low Temperature",
                                   f"{temp}°C → slow growth"))

            # =====================================================
            # 💧 IRRIGATION INTELLIGENCE (CORE FEATURE)
            # =====================================================

            irrigation_score = 0

            if ndvi is not None:
                if ndvi < 0.3:
                    irrigation_score += 2
                elif ndvi < 0.5:
                    irrigation_score += 1

            if temp is not None and temp > 32:
                irrigation_score += 2

            if humidity is not None and humidity < 40:
                irrigation_score += 1

            if weather and "rain" in weather:
                irrigation_score -= 3

            # 🔥 DECISION
            if irrigation_score >= 3:
                alerts.append(("💧 Irrigation Needed",
                               "High water stress detected. Irrigate now"))

            elif irrigation_score == 2:
                alerts.append(("💧 Irrigation Soon",
                               "Moderate stress. Plan irrigation"))

            elif irrigation_score <= 0:
                alerts.append(("🚫 No Irrigation Needed",
                               "Soil moisture likely sufficient"))

            # =====================================================
            # 🎯 PRIORITY FILTER (NO SPAM)
            # =====================================================

            alerts = alerts[:3]

            # =====================================================
            # 📊 FALLBACK (ALWAYS SEND SOMETHING)
            # =====================================================

            if not alerts:
                alerts.append((
                    "📊 Farm Summary",
                    f"NDVI: {ndvi}, Temp: {temp}, Weather: {weather}"
                ))

            # =====================================================
            # 🔔 SEND
            # =====================================================

            for title, body in alerts:

                message = messaging.Message(
                    notification=messaging.Notification(
                        title=title,
                        body=body,
                    ),
                    token=token,
                )

                messaging.send(message)
                sent += 1

            # =====================================================
            # 💾 SAVE STATE
            # =====================================================

            user_ref.set({
                "ndvi": ndvi,
                "last_updated": datetime.utcnow().isoformat()
            }, merge=True)

        except Exception as e:
            print("❌ Error:", e)

    return {"sent": sent}
