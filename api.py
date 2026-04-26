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
from firebase_admin import auth
from deep_translator import GoogleTranslator
from datetime import datetime


def get_alert_index():
    return datetime.utcnow().hour % 10

# 🔥 GLOBAL CACHE
translation_cache = {}

from deep_translator import GoogleTranslator

def translate_text(text, lang="bn"):
    if not text:
        return text

    # 🔥 NORMALIZE (CRITICAL)
    key = f"{text.lower().strip()}_{lang}"

    # 🔥 CACHE HIT
    if key in translation_cache:
        return translation_cache[key]

    try:
        translated = GoogleTranslator(
            source='auto',
            target=lang
        ).translate(text)

        # 🔥 SAVE CACHE
        translation_cache[key] = translated

        return translated

    except Exception as e:
        print("Translation error:", e)
        return text

import requests
def get_user_lang(user_id):
    try:
        doc = db.collection("farmers").document(user_id).get()
        data = doc.to_dict()
        return data.get("lang", "en") if data else "en"
    except:
        return "en"

translation_cache = {}

def translate_text(text, lang="bn"):
    key = f"{text}_{lang}"

    if key in translation_cache:
        return translation_cache[key]

    try:
        url = "https://translate.googleapis.com/translate_a/single"

        params = {
            "client": "gtx",
            "sl": "auto",
            "tl": lang,
            "dt": "t",
            "q": text
        }

        res = requests.get(url, params=params, timeout=5)
        translated = res.json()[0][0][0]

        translation_cache[key] = translated

        if len(translation_cache) > 1000:
            translation_cache.clear()

        return translated

    except:
        return text

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
    user_id: str | None = None
    lang: str = "en"


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

    # ✅ SAFE VALUES
    ndvi = data.ndvi if data.ndvi is not None else 0.45
    humidity = data.humidity if data.humidity is not None else 50
    rainfall = data.rainfall if data.rainfall is not None else 0
    temp = data.avg_temp if data.avg_temp is not None else 25

    try:
        df = pd.DataFrame([{
            "soil_type": (data.soil_type or "loamy").lower(),
            "fertilizer_type": (data.fertilizer_type or "urea").lower(),
            "crop_stage": (data.crop_stage or "vegetative").lower(),
            "stress_level": (data.stress_level or "low").lower(),
            "fertilizer_kg": data.fertilizer_kg or 0,
            "irrigation_count": data.irrigation_count or 0,
            "pesticide_sprays": data.pesticide_sprays or 0,
            "avg_temp": temp,
            "rainfall": rainfall,
            "humidity": humidity,
            "wind_speed": data.wind_speed or 2,
            "ndvi": ndvi
        }])

        # 🔥 CRITICAL: FIX COLUMN ORDER
        df = df[[
            "soil_type",
            "fertilizer_type",
            "crop_stage",
            "stress_level",
            "fertilizer_kg",
            "irrigation_count",
            "pesticide_sprays",
            "avg_temp",
            "rainfall",
            "humidity",
            "wind_speed",
            "ndvi"
        ]]

        # ✅ PREDICT
        y = float(yield_model.predict(df)[0])

        # 🔥 REAL CONFIDENCE (Random Forest ONLY)
        preds = [t.predict(df)[0] for t in yield_model.estimators_]
        std = float(pd.Series(preds).std())

        confidence = 100 - (std * 50)
        confidence = max(40, min(95, confidence))

    except Exception as e:
        return {"error": str(e)}

    # ✅ SAVE ONLY VALID
    if y > 0:
        add_yield_record(user_id, data.field_id, y)

    return {
        "predicted_yield": round(y, 2),
        "confidence": round(confidence, 1),
        "uncertainty": round(std, 3),
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
            ]

# 🔥 CLOSE POLYGON
            if coords[0] != coords[-1]:
                coords.append(coords[0])

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

        ndvi_img = latest_img.select("NDVI").updateMask(mask)
        ndwi_img = latest_img.select("NDWI").updateMask(mask)
        savi_img = latest_img.select("SAVI").updateMask(mask)
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

    # ✅ CORRECT PLACE
            lang = req.lang or "en"

            if lang != "en":
                trend_type = translate_text(trend_type, lang)

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

from fastapi import FastAPI, UploadFile, File, Form
import json, os, base64
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# 🔥 PROMPT
def build_prompt(crop, lang):
    if lang == "bn":
        return f"""
তুমি একজন কৃষি বিশেষজ্ঞ।

এই {crop} ফসলের পাতার ছবি বিশ্লেষণ করো।

শুধুমাত্র JSON ফরম্যাটে উত্তর দাও:
{{
  "disease": "...",
  "advice": "..."
}}

নিয়ম:
- সম্পূর্ণ বাংলা ভাষায় লিখবে
- সহজ কৃষক-বান্ধব ভাষা ব্যবহার করবে
- কীটনাশক বা সমাধান বলবে
"""
    else:
        return f"""
You are an agriculture expert.

Analyze this {crop} leaf image.

Give pesticide solution.

Return ONLY JSON:
{{
  "disease": "...",
  "advice": "..."
}}
"""


# 🔥 AI CALL
import re

def analyze_image(image_bytes, crop, lang):
    prompt = build_prompt(crop, lang)

    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    }
                ],
            }
        ],
    )

    text = response.choices[0].message.content

    print("RAW:", text)

    # 🔥 EXTRACT JSON ONLY
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    # 🔥 FALLBACK
    return {
        "disease": text.split("\n")[0],
        "advice": text
    }


# 🔥 API
@app.post("/predict-disease")
async def predict_disease(
    crop: str = Form(...),
    user_id: str = Form(...),
    file: UploadFile = File(...),
    lang: str = Form("en")
):
    print("LANG RECEIVED:", lang)

    try:
        contents = await file.read()

        result = analyze_image(contents, crop, lang)

        return {
            "crop": crop,
            "disease": result.get("disease", ""),
            "advice": result.get("advice", "")
        }

    except Exception as e:
        print("ERROR:", e)

        return {
            "disease": "অজানা" if lang == "bn" else "Unknown",
            "advice": "সার্ভার সমস্যা" if lang == "bn" else "Server error"
        }
    

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
from fastapi import Query

@app.get("/market-prices")
def get_prices(
    crop: str = Query(default=None),
    district: str = Query(default=None),
    sort: str = Query(default="price"),
    limit: int = Query(default=20),
    offset: int = Query(default=0),
    lang: str = Query(default="en")
):

    # 🔥 TRANSLATE SEARCH INPUT (BEFORE QUERY)
    if crop and lang != "en":
        crop = translate_text(crop, "en")

    if district and lang != "en":
        district = translate_text(district, "en")

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

    # 🔥 PAGINATION
    query += " LIMIT :limit OFFSET :offset"

    # 🔥 SINGLE QUERY EXECUTION
    with engine.connect() as conn:
        result = conn.execute(text(query), params)
        rows = [dict(r._mapping) for r in result]

    # 🔥 TRANSLATE OUTPUT
    if lang != "en":
        for r in rows:
            if r.get("commodity"):
                r["commodity"] = translate_text(r["commodity"], lang)
            if r.get("market"):
                r["market"] = translate_text(r["market"], lang)
            if r.get("district"):
                r["district"] = translate_text(r["district"], lang)

    return rows
#----------------------------------
# ================= FIREBASE =================

import os
import json
import firebase_admin
from firebase_admin import credentials, messaging

# 🔥 LOAD KEY
firebase_env = os.getenv("FIREBASE_KEY")

if not firebase_env:
    raise Exception("FIREBASE_KEY not set")

firebase_key = json.loads(firebase_env)
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
            if "Requested entity was not found" in str(e):
                db.collection("farmers").document(token).update({
                    "fcm_token": None
                })
                print("🧹 Removed invalid token")
            else:
                print("❌ Error:", e)


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

def get_weather(lat, lon, lang="en"):

    key = os.getenv("OPENWEATHER_API_KEY")

    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}&units=metric"

    try:
        res = requests.get(url, timeout=5).json()
        weather = res.get("weather", [{}])[0].get("main", "").lower()

        return t(weather, lang)

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
                "id": u.id,
                "token": d.get("fcm_token"),
                "lat": d.get("lat"),
                "lon": d.get("lon"),
            })

    return data

def build_24_notifications(lang, weather, temp, ndvi, forecast, news_list):

    alerts = []

    def t(en, bn):
        return bn if lang == "bn" else en

    # ================= 1. 🌧 RAIN TIMING =================
    for item in forecast.get("list", [])[:8]:  # next 24h
        cond = item.get("weather", [{}])[0].get("main", "").lower()

        if "rain" in cond:
            ts = item.get("dt_txt")
            time_str = ts.split()[1][:5]

            alerts.append((
                t("🌧 Rain Alert", "🌧 বৃষ্টি সতর্কতা"),
                t(f"Rain expected at {time_str}",
                  f"{time_str} সময় বৃষ্টি হতে পারে")
            ))
            break

    # ================= 2. 🚨 NDVI / CROP =================
    if ndvi is not None:
        if ndvi < 0.3:
            alerts.append((
                t("🚨 Crop Critical", "🚨 ফসল ঝুঁকিপূর্ণ"),
                t("Very low vegetation health",
                  "ফসলের অবস্থা খুব খারাপ")
            ))
        elif ndvi < 0.5:
            alerts.append((
                t("⚠️ Crop Moderate", "⚠️ মাঝারি ফসল"),
                t("Moderate crop condition",
                  "ফসলের অবস্থা মাঝারি")
            ))
        else:
            alerts.append((
                t("✅ Healthy Crop", "✅ ভালো ফসল"),
                t("Crop health is good",
                  "ফসলের অবস্থা ভালো")
            ))

    # ================= 3. 💧 IRRIGATION =================
    if temp and temp > 32:
        alerts.append((
            t("💧 Irrigation Needed", "💧 সেচ প্রয়োজন"),
            t("High temperature stress",
              "উচ্চ তাপমাত্রায় ফসলের চাপ বাড়ছে")
        ))

    # ================= 4. 🌾 YIELD =================
    if ndvi:
        yield_est = ndvi * 5

        alerts.append((
            t("🌾 Yield Forecast", "🌾 ফলন পূর্বাভাস"),
            t(f"Expected yield: {yield_est:.2f}",
              f"সম্ভাব্য ফলন: {yield_est:.2f}")
        ))

    # ================= 5. 🌤 WEATHER =================
    alerts.append((
        t("🌤 Weather Update", "🌤 আবহাওয়া আপডেট"),
        t(f"{weather}, {temp}°C",
          f"{weather}, {temp}°C")
    ))

    alerts.append((
        t("🌙 Tomorrow Planning", "🌙 আগামী দিনের পরিকল্পনা"),
        t("Prepare for tomorrow farming",
          "আগামী দিনের জন্য প্রস্তুত থাকুন")
    ))

    # ================= 6. 💰 MARKET =================
    alerts.append((
        t("💰 Market Price", "💰 বাজার মূল্য"),
        t("Check latest mandi prices",
          "আজকের বাজার মূল্য দেখুন")
    ))

    alerts.append((
        t("📈 Sell Opportunity", "📈 বিক্রির সুযোগ"),
        t("Prices may increase today",
          "আজ দাম বাড়তে পারে")
    ))

    # ================= 7. 📰 NEWS =================
    if news_list:
        for n in news_list[:2]:
            alerts.append((
                t("📰 Agri News", "📰 কৃষি সংবাদ"),
                n
            ))
    else:
        alerts.append((
            t("📰 Agri News", "📰 কৃষি সংবাদ"),
            t("Latest farming updates available",
              "নতুন কৃষি সংবাদ দেখুন")
        ))

    # ================= 8. 📊 ENGAGEMENT =================
    alerts += [
        (t("📊 NDVI Check", "📊 NDVI চেক করুন"),
         t("See your crop health map", "আপনার জমির অবস্থা দেখুন")),

        (t("📱 Open App", "📱 অ্যাপ খুলুন"),
         t("Check farm insights now", "এখনই ফসলের তথ্য দেখুন")),

        (t("🧠 Smart Tip", "🧠 স্মার্ট পরামর্শ"),
         t("AI can improve your yield", "AI ব্যবহার করে ফলন বাড়ান")),
    ]

    # ================= 9. 🌱 PRODUCTIVE TIPS =================
    tips = [
        ("Monitor crop daily", "নিয়মিত ফসল দেখুন"),
        ("Check soil moisture", "মাটির আর্দ্রতা পরীক্ষা করুন"),
        ("Apply fertilizer if needed", "প্রয়োজনে সার দিন"),
        ("Check pest attack", "পোকামাকড় দেখুন"),
        ("Update farm diary", "ফার্ম ডায়েরি আপডেট করুন"),
        ("Use satellite insights", "স্যাটেলাইট ডাটা ব্যবহার করুন"),
    ]

    for en, bn in tips:
        alerts.append((t("🌱 Tip", "🌱 পরামর্শ"), t(en, bn)))

    # ================= 10. FILL TO 24 =================
    i = 0
    while len(alerts) < 24:
        en, bn = tips[i % len(tips)]
        alerts.append((t("🌱 Tip", "🌱 পরামর্শ"), t(en, bn)))
        i += 1

    return alerts[:24]



@app.post("/smart-alerts")
def smart_alerts(data: dict):
    news_list = data.get("news", [])

    users = get_users()
    sent = 0
    print("TOTAL USERS:", len(users))
    #print("USER:", user_id, lat, lon, token)

    for u in users:
        try:
            user_id = u.get("id")
            if not user_id:
                print("❌ Missing user_id → skip")
                continue
            lang = get_user_lang(user_id)

            lat = u.get("lat")
            lon = u.get("lon")
            token = u.get("token")

            # 🔥 HARD FILTER
            if not token or lat is None or lon is None:
                continue

            now = datetime.utcnow() + timedelta(hours=5, minutes=30)  # convert to IST
            hour = now.hour

            # ================= WEATHER =================
            key = os.getenv("OPENWEATHER_API_KEY")

            weather = ""
            temp = None
            humidity = None

            try:
                weather_res = requests.get(
                    f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}&units=metric",
                    timeout=5
                ).json()

                weather = weather_res.get("weather", [{}])[0].get("main", "").lower()
                temp = weather_res.get("main", {}).get("temp")
                humidity = weather_res.get("main", {}).get("humidity")

            except Exception as e:
                print("Weather error:", e)

            # ================= NDVI =================
            try:
                ndvi = get_ndvi(lat, lon)
            except:
                ndvi = None

            # ================= FORECAST =================
            forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={key}&units=metric"
            try:
                forecast = requests.get(forecast_url, timeout=5).json()
            except:
                forecast = {}

            # ================= ALERTS =================
            alerts = build_24_notifications(
                lang, weather, temp, ndvi, forecast, news_list
            )

            # ================= TIME FILTER =================
            if 6 <= hour < 12:
                alerts = [a for a in alerts if (
                    "Weather" in a[0] or "Crop" in a[0] or
                    "আবহাওয়া" in a[0] or "ফসল" in a[0]
                )]

            elif 12 <= hour < 18:
                alerts = [a for a in alerts if (
                    "Irrigation" in a[0] or "Yield" in a[0] or
                    "সেচ" in a[0] or "ফলন" in a[0]
                )]

            elif 18 <= hour < 21:
                alerts = [a for a in alerts if (
                    "Market" in a[0] or "News" in a[0] or
                    "বাজার" in a[0] or "সংবাদ" in a[0]
                )]

            else:
                alerts = [a for a in alerts if (
                    "Tomorrow" in a[0] or "High temp" in a[0] or
                    "আগামীকাল" in a[0] or "তাপমাত্রা" in a[0]
                )]

            # ================= FALLBACK =================
            if not alerts:
                alerts = build_24_notifications(
                    user_id, lang, weather, temp, humidity, ndvi, news_list
                )

            # ================= COOLDOWN =================
            user_ref = db.collection("alerts_state").document(token)
            prev = user_ref.get().to_dict() or {}

            now = datetime.utcnow()
            last_sent = prev.get("last_sent")

            if last_sent:
                last_time = datetime.fromisoformat(last_sent)
                if now - last_time < timedelta(minutes=55):
                    print("⛔ Cooldown active → skip user")
                    continue

            # ================= SEND =================
            # ================= SEND =================
            if not alerts:
                continue

            # 12 AM – 4 AM window
            if hour < 0 or hour > 4:
                continue

            slot = hour  # 0–4 directly

# 🔥 LOAD PREVIOUS STATE
            user_ref = db.collection("alerts_state").document(token)
            prev = user_ref.get().to_dict() or {}

            prev_index = prev.get("last_index", -1)

            import random

# 🔥 ROTATION LOGIC
            index = (prev_index + 1) % len(alerts)

# 🔥 RANDOM BOOST (30%)
            if random.random() < 0.3:
                index = random.randint(0, len(alerts) - 1)

# 🔥 PRIORITY OVERRIDE
            priority = [a for a in alerts if "Rain" in a[0] or "বৃষ্টি" in a[0]]

            if priority:
                title, body = priority[0]
            else:
                title, body = alerts[index]

# 🔥 SEND
            message = messaging.Message(
                notification=messaging.Notification(
                    title=title,
                    body=body,
                ),
                token=token,
            )

            messaging.send(message)
            sent += 1

# 🔥 SAVE STATE
            user_ref.set({
                "ndvi": ndvi,
                "last_sent": datetime.utcnow().isoformat(),
                "last_index": index
            }, merge=True)

            # ================= SAVE =================

        except Exception as e:
            print("❌ Error:", e)
            continue

    return {"sent": sent}

#=================DELETE ACCOUNT==============================


from fastapi import Depends, Header

@app.delete("/delete-account")
async def delete_account(authorization: str = Header(...)):

    try:
        # 🔐 VERIFY FIREBASE TOKEN
        token = authorization.replace("Bearer ", "")
        decoded = auth.verify_id_token(token)

        user_id = decoded["uid"]

        # 🔥 DELETE FIREBASE USER
        auth.delete_user(user_id)

        # 🔥 DELETE FIRESTORE DATA
        db.collection("farmers").document(user_id).delete()
        db.collection("alerts_state").document(user_id).delete()

        # TODO: delete yield history if stored separately

        return {"success": True, "message": "Account deleted"}

    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


from fastapi.responses import HTMLResponse

@app.get("/delete-account-info", response_class=HTMLResponse)
async def delete_account_info():
    return """
    <html>
    <head>
        <title>KishanSeva Account Deletion</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
    </head>
    <body style="font-family: Arial; padding:20px; background:#F5F7F2;">

        <h2 style="color:#689F38;">KishanSeva Account Deletion</h2>

        <p>You can delete your account directly inside the app:</p>

        <ol>
            <li>Open KishanSeva</li>
            <li>Go to Profile</li>
            <li>Tap <b>"Delete My Account"</b></li>
        </ol>

        <p>If you cannot access the app, contact support:</p>
        <p><b>Email:</b> contactkishanseva@gmail.com</p>

        <h4>Data that will be deleted:</h4>
        <ul>
            <li>User account</li>
            <li>Farm field data</li>
            <li>NDVI and advisory data</li>
        </ul>

        <p><b>Processing time:</b> Within 7 days</p>

    </body>
    </html>
    """

    
@app.post("/translate")
def translate_api(data: dict):
    text = data.get("text") or ""
    lang = data.get("lang", "bn")

    translated = translate_text(text, lang)

    return {"translated": translated}

@app.get("/health")
def health():
    return {"status": "ok"}
