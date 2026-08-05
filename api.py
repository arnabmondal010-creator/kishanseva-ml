# api.py
# -*- coding: utf-8 -*-

from builtins import round
import os
import json
import ee
import pandas as pd
import joblib
import feedparser
import hmac
import hashlib
import asyncio

from fastapi import (
    FastAPI,
    HTTPException,
    Response,
    Depends,
    Header,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import razorpay
import traceback

from limits import can_use, get_user_plan, set_user_plan, mark_used
from ai_service import analyze_image
from services.yield_history_service import add_yield_record, get_history
from functools import lru_cache
import time
from firebase_admin import auth
from deep_translator import GoogleTranslator
from datetime import datetime
from datetime import timedelta
from datetime import datetime, timezone, timedelta

def to_bengali_number(text):
    en = "0123456789"
    bn = "০১২৩৪৫৬৭৮৯"
    table = str.maketrans(en, bn)
    return str(text).translate(table)

def to_hindi_number(text):
    en = "0123456789"
    hi = "०१२३४५६७८९"
    table = str.maketrans(en, hi)
    return str(text).translate(table)

def get_yield_prediction(ndvi, temp, humidity):
    try:
        url = "https://kishanseva-ai.onrender.com/predict-yield"

        payload = {
            "ndvi": ndvi or 0.5,
            "avg_temp": temp or 28,
            "humidity": humidity or 60,
            "rainfall": 0
        }

        res = requests.post(url, json=payload, timeout=3).json()

        return res.get("predicted_yield")
    except:
        return None
    
def get_time_period(dt, lang):

    hour = dt.hour

    # Bengali
    if lang == "bn":

        if 4 <= hour < 10:
            return "সকাল"

        elif 10 <= hour < 14:
            return "দুপুর"

        elif 14 <= hour < 17:
            return "বিকাল"

        elif 17 <= hour < 20:
            return "সন্ধ্যা"

        else:
            return "রাত"

    # Hindi
    elif lang == "hi":

        if 4 <= hour < 10:
            return "सुबह"

        elif 10 <= hour < 14:
            return "दोपहर"

        elif 14 <= hour < 17:
            return "शाम"

        elif 17 <= hour < 20:
            return "संध्या"

        else:
            return "रात"

    # English
    else:

        if 4 <= hour < 10:
            return "morning"

        elif 10 <= hour < 14:
            return "noon"

        elif 14 <= hour < 17:
            return "afternoon"

        elif 17 <= hour < 20:
            return "evening"

        else:
            return "night"
    
def get_market_price():
    try:
        url = "https://kishanseva-ai.onrender.com/market-prices?limit=1"
        res = requests.get(url, timeout=3).json()

        if res:
            r = res[0]
            return f"{r['commodity']} ₹{r['price']}"
    except:
        return None

    return None


def get_agri_news(lang="en"):
    feeds_bn = [
        "https://www.anandabazar.com/rss",
        "https://news.google.com/rss/search?q=কৃষি&hl=bn&gl=IN&ceid=IN:bn"
    ]

    feeds_en = [
        "https://agricoop.nic.in/en/rss.xml",
        "https://news.google.com/rss/search?q=agriculture+india&hl=en-IN&gl=IN&ceid=IN:en"
    ]

    feeds = feeds_bn if lang == "bn" else feeds_en

    news = []

    for url in feeds:
        try:
            d = feedparser.parse(url)
            for e in d.entries[:2]:
                news.append(e.title)
        except:
            continue

    return news[:3]


def get_irrigation(temp, humidity, ndvi):
    try:
        url = "https://kishanseva-ai.onrender.com/predict-irrigation"

        payload = {
            "soil": "loamy",
            "crop": "rice",
            "temperature": temp or 30,
            "humidity": humidity or 60,
            "rainfall": 0,
            "ndvi": ndvi or 0.5,
            "infiltration": 0.5
        }

        res = requests.post(url, json=payload, timeout=3).json()

        return res.get("irrigation_mm")
    except:
        return None


def get_alert_index():
    return datetime.utcnow().hour % 10

# 🔥 GLOBAL CACHE
translation_cache = {}

from deep_translator import GoogleTranslator

# def translate_text(text, lang="bn"):
  #  if not text:
 #       return text
#
 #   # 🔥 NORMALIZE (CRITICAL)
#    key = f"{text.lower().strip()}_{lang}"

  #  # 🔥 CACHE HIT
 #   if key in translation_cache:
 #       return translation_cache[key]
#
 #   try:
#        translated = GoogleTranslator(
   #         source='auto',
  #          target=lang
 #       ).translate(text)
#
 #       # 🔥 SAVE CACHE
#        translation_cache[key] = translated

   #     return translated

  ##  except Exception as e:
    ##    print("Translation error:", e)
   ##     return text

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

        res = requests.get(url, params=params, timeout=3)
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
    field_id: str | None = "default"   # 🔥 ADD THIS
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
        key = f"{req.user_id or 'guest'}_{req.field_id or 'default'}"
        if key in NDVI_CACHE:
            data, ts = NDVI_CACHE[key]
            if time.time() - ts < CACHE_TTL:
                return data

        boundary = req.boundary
        # 🔥 FETCH FROM DB FIRST (CRITICAL)
        if (not boundary or len(boundary) < 3) and req.user_id:
            try:
                field_id = getattr(req, "field_id", "default")

                doc = db.collection("fields")\
                .document(req.user_id)\
                .collection("user_fields")\
                .document(field_id)\
                .get()

                if doc.exists:
                    boundary = doc.to_dict().get("boundary")
                print("BOUNDARY FROM DB:", boundary)
            except:
                boundary = None

        try:
            if isinstance(boundary, str):
                boundary = json.loads(boundary)
        except:
            boundary = None

        # ================= GEOMETRY =================
        if isinstance(boundary, list) and len(boundary) >= 3:
            if req.user_id:
                field_id = getattr(req, "field_id", "default")

                db.collection("fields")\
                    .document(req.user_id)\
                    .collection("user_fields")\
                    .document(field_id)\
                    .set({
                        "boundary": boundary
                    }, merge=True)

            coords = [
                [float(p["lon"]), float(p["lat"])]
                for p in boundary
            ]

# 🔥 CLOSE POLYGON
            if coords[0] != coords[-1]:
                coords.append(coords[0])

            geom = ee.Geometry.Polygon([coords])

        else:
            return {
                "error": "Boundary missing. Cannot generate polygon heatmap."
            }
        print("BOUNDARY RECEIVED:", req.boundary)

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
        # 🔥 SAVE NDVI TO FIRESTORE
        if req.user_id:
            try:
                field_id = getattr(req, "field_id", "default")

                db.collection("fields")\
                    .document(req.user_id)\
                    .collection("user_fields")\
                    .document(field_id)\
                    .set({
                        "ndvi": latest["ndvi"],
                        "ndvi_updated": datetime.utcnow().isoformat()
                }, merge=True)
            except Exception as e:
                print("NDVI save error:", e)

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

    # Bengali
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

    # Hindi
    elif lang == "hi":

        return f"""
तुम एक कृषि विशेषज्ञ हो।

इस {crop} फसल की पत्ती की तस्वीर का विश्लेषण करो।

सिर्फ JSON फॉर्मेट में उत्तर दो:
{{
  "disease": "...",
  "advice": "..."
}}

नियम:
- पूरा उत्तर हिंदी में होना चाहिए
- आसान किसान-हितैषी भाषा का उपयोग करो
- बीमारी और समाधान स्पष्ट बताओ
- कीटनाशक या उपचार भी बताओ
"""

    # English
    else:

        return f"""
You are an agriculture expert.

Analyze this {crop} crop leaf image.

Return ONLY valid JSON:
{{
  "disease": "...",
  "advice": "..."
}}

Rules:
- Respond in English
- Use farmer-friendly language
- Mention treatment and pesticide if needed
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
            "disease":
                "অজানা" if lang == "bn"
        else "अज्ञात" if lang == "hi"
        else "Unknown",

             "advice":
                "সার্ভার সমস্যা" if lang == "bn"
        else "सर्वर समस्या" if lang == "hi"
        else "Server error"
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
# ================= RAZORPAY =================

RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")

if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
    raise Exception("Razorpay credentials not configured")

razorpay_client = razorpay.Client(
    auth=(
        RAZORPAY_KEY_ID,
        RAZORPAY_KEY_SECRET,
    )
)

print("Razorpay initialized")
RAZORPAY_WEBHOOK_SECRET = os.getenv(
    "RAZORPAY_WEBHOOK_SECRET"
)

if not RAZORPAY_WEBHOOK_SECRET:
    raise Exception(
        "Razorpay webhook secret not configured"
    )


from typing import Optional

class CreateRazorpayOrderRequest(BaseModel):
    listingId: Optional[str] = None
    orderId: Optional[str] = None
    checkoutId: Optional[str] = None     # Cart Checkout
class VerifyRazorpayPaymentRequest(BaseModel):
    razorpay_payment_id: str
    razorpay_order_id: str
    razorpay_signature: str

    listingId: str | None = None
    checkoutId: Optional[str] = None
    orderId: str | None = None

class VerifyPickupOtpRequest(BaseModel):
    orderId: str
    otp: str
from typing import List

class CheckoutItem(BaseModel):
    listingId: str
    quantity: float


class CreateCheckoutRequest(BaseModel):
    addressId: str
    deliveryMethod: str
    subtotal: float
    deliveryCharge: float
    grandTotal: float
    items: List[CheckoutItem]


def verify_firebase_token(
    authorization: str = Header(...),
):
    try:
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Invalid authorization header",
            )

        token = authorization.split(
            "Bearer ",
            1,
        )[1]

        decoded_token = auth.verify_id_token(token)

        return decoded_token

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired authentication token",
        )
@app.post("/checkout/create")
async def create_checkout(
    request: CreateCheckoutRequest,
    user=Depends(verify_firebase_token),
):

    buyer_id = user["uid"]

    if not request.items:
        raise HTTPException(
            status_code=400,
            detail="Cart is empty",
        )

    checkout_ref = (
        db.collection("commerce_checkouts")
        .document()
    )

    checkout_id = checkout_ref.id

    subtotal = 0.0
    total_weight = 0.0
    seller_totals = {}
    checkout_items = []

    for item in request.items:

        # First try farm listings
        listing_doc = (
            db.collection("commerce_listings")
            .document(item.listingId)
            .get()
        )

        product_type = "farm"

# If not found, try retail products
        if not listing_doc.exists:

            listing_doc = (
                db.collection("commerce_shop_products")
                .document(item.listingId)
                .get()
            )

            product_type = "retail"

        if not listing_doc.exists:

            raise HTTPException(
                status_code=404,
                detail=f"Product {item.listingId} not found",
            )

        listing = listing_doc.to_dict()

        if product_type == "farm":

            available_qty = float(
                listing.get("quantity", 0)
            )

        else:

            available_qty = float(
                listing.get("stock", 0)
            )

        if item.quantity <= 0:

            raise HTTPException(
                status_code=400,
                detail="Invalid quantity",
            )

        product_name = (
            listing.get("cropName")
            or listing.get("productName")
            or "Product"
        )

        if item.quantity > available_qty:

            raise HTTPException(
                status_code=400,
                detail=f"{product_name} is out of stock",
            )

        price = float(
            listing.get("pricePerUnit", 0)
        )

        item_total = price * item.quantity

        subtotal += item_total

        weight_per_unit = float(
            listing.get("weightPerUnit", 1)
        )

        total_weight += (
            weight_per_unit * item.quantity
        )

        seller_id = listing["sellerId"]

        seller_totals[seller_id] = (
            seller_totals.get(seller_id, 0)
            + item_total
        )

        checkout_items.append({

            "listingId": item.listingId,

            "sellerId": seller_id,
            "productType": product_type,

            "productName": product_name,

            "quantity": item.quantity,

            "pricePerUnit": price,

            "totalPrice": item_total,

            "weightPerUnit": weight_per_unit,

        })
        address_ref = (
        db.collection("commerce_addresses")
        .document(request.addressId)
    )

    address_doc = address_ref.get()

    if not address_doc.exists:

        raise HTTPException(
            status_code=404,
            detail="Delivery address not found",
        )

    address = address_doc.to_dict()

    if address["buyerId"] != buyer_id:

        raise HTTPException(
            status_code=403,
            detail="Address does not belong to this buyer",
        )
    checkout_data = {

        "checkoutId": checkout_id,

        "buyerId": buyer_id,

        "status": "pending_payment",

        "addressId": request.addressId,

        "deliveryMethod": request.deliveryMethod,

        "address": address,

        "subtotal": subtotal,

        "deliveryCharge": request.deliveryCharge,

        "grandTotal": request.grandTotal,

        "totalWeight": total_weight,

        "sellerTotals": seller_totals,

        "items": checkout_items,

        "createdAt": firestore.SERVER_TIMESTAMP,

        "updatedAt": firestore.SERVER_TIMESTAMP,

    }

    checkout_ref.set(checkout_data)
    return {

        "success": True,

        "checkoutId": checkout_id,

        "subtotal": subtotal,

        "deliveryCharge": request.deliveryCharge,

        "grandTotal": request.grandTotal,

    }


@app.post("/payments/razorpay/create-order")
async def create_razorpay_order(
    request: CreateRazorpayOrderRequest,
    user=Depends(verify_firebase_token),
):
    try:
        # Firebase UID from verified token
        buyer_id = user["uid"]

# -----------------------------
# CART CHECKOUT
# -----------------------------
        if request.checkoutId:

            checkout_ref = (
                db.collection("commerce_checkouts")
                .document(request.checkoutId)
            )

            checkout_doc = checkout_ref.get()

            if not checkout_doc.exists:
                raise HTTPException(
                    status_code=404,
                    detail="Checkout not found",
                )

            checkout = checkout_doc.to_dict()

            if checkout["buyerId"] != buyer_id:
                raise HTTPException(
                    status_code=403,
                    detail="Unauthorized",
                )

            if checkout.get("paymentStatus") == "paid":
                raise HTTPException(
                    status_code=400,
                    detail="Checkout already paid",
                )

            amount = float(checkout["grandTotal"])
            amount_paise = int(round(amount * 100))

            razorpay_order = razorpay_client.order.create(
                data={
                    "amount": amount_paise,
                    "currency": "INR",
                    "receipt": f"cart_{request.checkoutId}",
                    "notes": {
                        "checkoutId": request.checkoutId,
                        "buyerId": buyer_id,
                        "type": "cart",
                    },
                }
            )

            return {
                "success": True,
                "keyId": RAZORPAY_KEY_ID,
                "amount": amount_paise,
                "currency": "INR",
                "razorpayOrderId": razorpay_order["id"],
                "checkoutId": request.checkoutId,
            }

# -----------------------------
# AUCTION PAYMENT
# -----------------------------
        if request.orderId:

            order_ref = db.collection(
                "commerce_orders"
            ).document(request.orderId)

            order_doc = order_ref.get()

            if not order_doc.exists:
                raise HTTPException(
                    status_code=404,
                    detail="Order not found",
                )

            order = order_doc.to_dict()

            if order["buyerId"] != buyer_id:
                raise HTTPException(
                    status_code=403,
                    detail="Unauthorized",
                )

            if order.get("paymentStatus") == "paid":
                raise HTTPException(
                    status_code=400,
                    detail="Already paid",
                )

            price = float(order["orderAmount"])

            amount_paise = int(price * 100)

            razorpay_order = razorpay_client.order.create(
                data={
                    "amount": amount_paise,
                    "currency": "INR",
                    "receipt": f"auction_{request.orderId}",
                    "notes": {
                        "orderId": request.orderId,
                        "buyerId": buyer_id,
                        "type": "auction",
                    },
                }
            )

            return {
                "success": True,
                "keyId": RAZORPAY_KEY_ID,
                "amount": amount_paise,
                "currency": "INR",
                "razorpayOrderId": razorpay_order["id"],
                "orderId": request.orderId,
            }

# -----------------------------
# INSTANT BUY
# -----------------------------

        if not request.listingId:
            raise HTTPException(
                status_code=400,
                detail="listingId required",
            )

        listing_ref = db.collection(
            "commerce_listings"
        ).document(request.listingId)

        listing_doc = listing_ref.get()

        if not listing_doc.exists:
            raise HTTPException(
                status_code=404,
                detail="Listing not found",
            )

        listing = listing_doc.to_dict()

        # Check listing status
        if listing.get("status") != "active":
            raise HTTPException(
                status_code=400,
                detail="Listing is not active",
            )

        # Check already sold
        if listing.get("sold", False) is True:
            raise HTTPException(
                status_code=400,
                detail="Listing already sold",
            )

        # Check Instant Buy
        if listing.get("instantBuyEnabled", False) is not True:
            raise HTTPException(
                status_code=400,
                detail="Instant Buy is not available",
            )

        # Server-controlled price
        price = listing.get("instantBuyPrice")

        if price is None:
            raise HTTPException(
                status_code=400,
                detail="Instant Buy price missing",
            )

        try:
            price = float(price)

        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="Invalid Instant Buy price",
            )

        if price <= 0:
            raise HTTPException(
                status_code=400,
                detail="Invalid Instant Buy price",
            )

        # Convert rupees to paise
        amount_paise = int(round(price * 100))

        # Create Razorpay order
        razorpay_order = razorpay_client.order.create(
            data={
                "amount": amount_paise,
                "currency": "INR",
                "receipt": f"ks_{request.listingId[:25]}",
                "notes": {
                    "listingId": request.listingId,
                    "buyerId": buyer_id,
                },
            }
        )

        return {
            "success": True,
            "razorpayOrderId": razorpay_order["id"],
            "keyId": RAZORPAY_KEY_ID,
            "amount": razorpay_order["amount"],
            "currency": razorpay_order["currency"],
            "listingId": request.listingId,
        }

    except HTTPException:
        raise

    except Exception as e:
        print(
            "RAZORPAY CREATE ORDER ERROR:",
            str(e),
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to create payment order",
        )
def find_existing_razorpay_refund(
    payment_id: str,
):
    """
    Check Razorpay for an existing refund
    associated with this payment.

    Returns the latest matching refund,
    or None if no refund exists.
    """

    try:
        response = (
            razorpay_client
            .payment
            .fetch_multiple_refund(
                payment_id
            )
        )

        items = (
            response.get("items", [])
            if isinstance(response, dict)
            else []
        )

        if not items:
            return None

        # Prefer the most recently created refund.

        items.sort(
            key=lambda item:
                item.get("created_at", 0),
            reverse=True,
        )

        return items[0]

    except Exception as e:

        print(
            "RAZORPAY REFUND RECONCILIATION ERROR:",
            payment_id,
            str(e),
        )

        raise
def refund_failed_instant_buy(
    payment_id: str,
    razorpay_order_id: str,
    listing_id: str,
    buyer_id: str,
    reason: str,
):
    """
    Idempotently initiate a full refund for an
    Instant Buy payment that cannot be fulfilled.

    Firestore document:
    commerce_refund_locks/{payment_id}
    """

    # ==========================================
    # 1. VALIDATE REQUIRED DATA
    # ==========================================

    if not payment_id:
        raise Exception(
            "Missing payment ID for refund"
        )

    if not razorpay_order_id:
        raise Exception(
            "Missing Razorpay order ID for refund"
        )

    if not listing_id:
        raise Exception(
            "Missing listing ID for refund"
        )

    if not buyer_id:
        raise Exception(
            "Missing buyer ID for refund"
        )

    refund_lock_ref = (
        db.collection(
            "commerce_refund_locks"
        )
        .document(
            payment_id
        )
    )

    # ==========================================
    # 2. FAST IDEMPOTENCY CHECK
    # ==========================================

    existing_lock = (
        refund_lock_ref.get()
    )

    if existing_lock.exists:

        data = (
            existing_lock.to_dict()
            or {}
        )

        status = data.get(
            "status"
        )

        # ======================================
        # REFUND ALREADY SUBMITTED / COMPLETED
        # ======================================

        if status in [
            "requested",
            "pending",
            "processed",
        ]:

            return {
                "success":
                    True,

                "alreadyRequested":
                    True,

                "refundId":
                    data.get(
                        "refundId"
                    ),

                "status":
                    status,
            }

        # ======================================
        # ANOTHER WORKER IS PROCESSING
        # ======================================

        if status == "processing":

            processing_started_at = (
                data.get(
                    "processingStartedAt"
                )
            )

            stale = False

            if processing_started_at:

                try:

                    now = datetime.now(
                        timezone.utc
                    )

                    stale = (
                        now
                        - processing_started_at
                        > timedelta(
                            minutes=10
                        )
                    )

                except Exception:
                    stale = False

            # Active processing lock.
            # Do not request another refund.

            if not stale:

                return {
                    "success":
                        True,

                    "alreadyRequested":
                        True,

                    "refundId":
                        data.get(
                            "refundId"
                        ),

                    "status":
                        "processing",
                }

            # ======================================
            # STALE REFUND LOCK
            # ======================================

            existing_refund = (
                find_existing_razorpay_refund(
                    payment_id
                )
            )

            # Previous refund actually reached
            # Razorpay. Reconcile local state.

            if existing_refund:

                existing_refund_id = (
                    existing_refund.get(
                        "id"
                    )
                )

                existing_refund_status = (
                    existing_refund.get(
                        "status"
                    )
                    or "requested"
                )

                refund_lock_ref.set(
                    {
                        "refundId":
                            existing_refund_id,

                        "status":
                            existing_refund_status,

                        "razorpayRefundStatus":
                            existing_refund_status,

                        "reconciled":
                            True,

                        "reconciledAt":
                            firestore.SERVER_TIMESTAMP,

                        "updatedAt":
                            firestore.SERVER_TIMESTAMP,
                    },
                    merge=True,
                )

                return {
                    "success":
                        True,

                    "alreadyRequested":
                        True,

                    "refundId":
                        existing_refund_id,

                    "status":
                        existing_refund_status,
                }

            # No refund found at Razorpay.
            # Recover stale lock and allow
            # atomic claim below to retry.

            refund_lock_ref.set(
                {
                    "status":
                        "failed",

                    "failureReason":
                        "Stale processing lock recovered",

                    "reconciled":
                        True,

                    "reconciledAt":
                        firestore.SERVER_TIMESTAMP,

                    "updatedAt":
                        firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )
    # ==========================================
    # 3. ATOMICALLY CLAIM REFUND
    # ==========================================

    transaction = (
        db.transaction()
    )

    @firestore.transactional
    def claim_refund(
        transaction,
    ):

        lock_doc = transaction.get(refund_lock_ref)

        if hasattr(lock_doc, "__next__"):
            lock_doc = next(lock_doc)

        if lock_doc.exists:

            lock_data = (
                lock_doc.to_dict()
                or {}
            )

            # Verify that an existing lock
            # belongs to the same payment context.

            locked_order_id = (
                lock_data.get(
                    "razorpayOrderId"
                )
            )

            locked_listing_id = (
                lock_data.get(
                    "listingId"
                )
            )

            locked_buyer_id = (
                lock_data.get(
                    "buyerId"
                )
            )

            if (
                locked_order_id
                and
                locked_order_id
                != razorpay_order_id
            ):
                raise Exception(
                    "Refund lock order mismatch"
                )

            if (
                locked_listing_id
                and
                locked_listing_id
                != listing_id
            ):
                raise Exception(
                    "Refund lock listing mismatch"
                )

            if (
                locked_buyer_id
                and
                locked_buyer_id
                != buyer_id
            ):
                raise Exception(
                    "Refund lock buyer mismatch"
                )

            current_status = (
                lock_data.get(
                    "status"
                )
            )

            if current_status in [
                "processing",
                "requested",
                "pending",
                "processed",
            ]:

                return {
                    "claimed":
                        False,

                    "refundId":
                        lock_data.get(
                            "refundId"
                        ),

                    "status":
                        current_status,
                }

        # This transaction gives this worker
        # ownership of initiating the refund.

        transaction.set(
            refund_lock_ref,
            {
                "paymentId":
                    payment_id,

                "razorpayOrderId":
                    razorpay_order_id,

                "listingId":
                    listing_id,

                "buyerId":
                    buyer_id,

                "reason":
                    reason,

                "status":
                    "processing",

                "processingStartedAt":
                    firestore.SERVER_TIMESTAMP,

                "updatedAt":
                    firestore.SERVER_TIMESTAMP,

                "createdAt":
                    firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

        return {
            "claimed":
                True,

            "refundId":
                None,

            "status":
                "processing",
        }

    claim_result = (
        claim_refund(
            transaction
        )
    )

    # ==========================================
    # 4. ANOTHER WORKER ALREADY CLAIMED IT
    # ==========================================

    if not claim_result.get(
        "claimed",
        False,
    ):

        return {
            "success":
                True,

            "alreadyRequested":
                True,

            "refundId":
                claim_result.get(
                    "refundId"
                ),

            "status":
                claim_result.get(
                    "status"
                ),
        }

    # ==========================================
    # 5. VERIFY PAYMENT WITH RAZORPAY
    # ==========================================

    try:

        payment = (
            razorpay_client
            .payment
            .fetch(
                payment_id
            )
        )

        if (
            payment.get(
                "order_id"
            )
            != razorpay_order_id
        ):
            raise Exception(
                "Refund payment order mismatch"
            )

        payment_status = (
            payment.get(
                "status"
            )
        )

        # ======================================
        # ALREADY REFUNDED
        # ======================================

        if payment_status == "refunded":

            refund_lock_ref.set(
                {
                    "status":
                        "processed",

                    "updatedAt":
                        firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )

            return {
                "success":
                    True,

                "alreadyRequested":
                    True,

                "refundId":
                    None,

                "status":
                    "processed",
            }

        # ======================================
        # PAYMENT MUST BE CAPTURED
        # ======================================

        if payment_status != "captured":

            raise Exception(
                f"Payment cannot be refunded "
                f"in status: {payment_status}"
            )

        # ==========================================
        # 6. REQUEST FULL REFUND FROM RAZORPAY
        # ==========================================

        refund = (
            razorpay_client
            .payment
            .refund(
                payment_id,
                {}
            )
        )

        refund_id = (
            refund.get(
                "id"
            )
        )

        refund_status = (
            refund.get(
                "status"
            )
            or "requested"
        )

        # ==========================================
        # 7. SAVE RAZORPAY REFUND RESULT
        # ==========================================

        refund_lock_ref.set(
            {
                "refundId":
                    refund_id,

                "status":
                    refund_status,

                "razorpayRefundStatus":
                    refund_status,

                "refundRequestedAt":
                    firestore.SERVER_TIMESTAMP,
                "processingCompletedAt":
                    firestore.SERVER_TIMESTAMP,

                "updatedAt":
                    firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

        return {
            "success":
                True,

            "alreadyRequested":
                False,

            "refundId":
                refund_id,

            "status":
                refund_status,
        }

    except Exception as refund_error:

        # ==========================================
        # 8. MARK CLAIM AS FAILED
        # ==========================================

        refund_lock_ref.set(
            {
                "status":
                    "failed",

                "error":
                    str(
                        refund_error
                    )[:500],

                "failedAt":
                    firestore.SERVER_TIMESTAMP,

                "updatedAt":
                    firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

        raise

def finalize_instant_buy(
    razorpay_order_id: str,
    payment_id: str,
    listing_id: str,
    buyer_id: str,
):
    """
    Idempotently finalize an Instant Buy payment.

    Uses:
    commerce_payment_locks/{razorpay_order_id}

    Both /verify and /webhook can safely call this function.
    """

    # ==========================================
    # 1. VALIDATE INPUT
    # ==========================================

    if not razorpay_order_id:
        raise Exception(
            "Missing Razorpay order ID"
        )

    if not payment_id:
        raise Exception(
            "Missing Razorpay payment ID"
        )

    if not listing_id:
        raise Exception(
            "Missing listing ID"
        )

    if not buyer_id:
        raise Exception(
            "Missing buyer ID"
        )

    # ==========================================
    # 2. DEFINE FIRESTORE REFERENCES
    # ==========================================

    listing_ref = (
        db.collection(
            "commerce_listings"
        )
        .document(
            listing_id
        )
    )

    payment_lock_ref = (
        db.collection(
            "commerce_payment_locks"
        )
        .document(
            razorpay_order_id
        )
    )

    # ==========================================
    # 3. FAST IDEMPOTENCY CHECK
    # ==========================================

    existing_lock = (
        payment_lock_ref.get()
    )

    if existing_lock.exists:

        lock_data = (
            existing_lock.to_dict()
            or {}
        )

        if (
            lock_data.get("status")
            == "finalized"
            and lock_data.get(
                "commerceOrderId"
            )
        ):
            return {
                "success":
                    True,

                "alreadyProcessed":
                    True,

                "orderId":
                    lock_data[
                        "commerceOrderId"
                    ],

                "orderStatus":
                    "confirmed",
            }

    # ==========================================
    # 4. FETCH PAYMENT FROM RAZORPAY
    # ==========================================

    payment = (
        razorpay_client
        .payment
        .fetch(
            payment_id
        )
    )

    if (
        payment.get("order_id")
        != razorpay_order_id
    ):
        raise Exception(
            "Payment order mismatch"
        )

    if payment.get("status") not in [
        "authorized",
        "captured",
    ]:
        raise Exception(
            "Payment is not successful"
        )

    # ==========================================
    # 5. FETCH RAZORPAY ORDER
    # ==========================================

    razorpay_order = (
        razorpay_client
        .order
        .fetch(
            razorpay_order_id
        )
    )

    notes = (
        razorpay_order.get("notes")
        or {}
    )

    if (
        notes.get("listingId")
        != listing_id
    ):
        raise Exception(
            "Listing mismatch"
        )

    if (
        notes.get("buyerId")
        != buyer_id
    ):
        raise Exception(
            "Buyer mismatch"
        )

    # ==========================================
    # 6. FETCH LISTING FOR AMOUNT CHECK
    # ==========================================

    listing_doc = (
        listing_ref.get()
    )

    if not listing_doc.exists:
        raise Exception(
            "Listing not found"
        )

    listing = (
        listing_doc.to_dict()
        or {}
    )

    try:
        price = float(
            listing.get(
                "instantBuyPrice",
                0,
            )
        )

    except (
        TypeError,
        ValueError,
    ):
        raise Exception(
            "Invalid Instant Buy price"
        )

    if price <= 0:
        raise Exception(
            "Invalid Instant Buy price"
        )

    expected_amount = int(
        round(
            price * 100
        )
    )

    if (
        int(
            payment.get(
                "amount",
                0,
            )
        )
        != expected_amount
    ):
        raise Exception(
            "Payment amount mismatch"
        )

    if (
        int(
            razorpay_order.get(
                "amount",
                0,
            )
        )
        != expected_amount
    ):
        raise Exception(
            "Razorpay order amount mismatch"
        )

    # ==========================================
    # 7. GENERATE ORDER ID + PICKUP OTP
    # ==========================================

    import secrets

    pickup_otp = str(
        secrets.randbelow(
            900000
        )
        + 100000
    )

    order_ref = (
        db.collection(
            "commerce_orders"
        )
        .document()
    )

    # ==========================================
    # 8. FIRESTORE TRANSACTION
    # ==========================================

    transaction = (
        db.transaction()
    )

    @firestore.transactional
    def finalize_transaction(
        transaction,
    ):

        # IMPORTANT:
        # All transactional reads happen
        # before transactional writes.

        current_lock_doc = transaction.get(payment_lock_ref)

        if hasattr(current_lock_doc, "__next__"):
            current_lock_doc = next(current_lock_doc)

        current_listing_doc = transaction.get(listing_ref)

        if hasattr(current_listing_doc, "__next__"):
            current_listing_doc = next(current_listing_doc)

        

        # ======================================
        # PAYMENT LOCK ALREADY EXISTS
        # ======================================

        if current_lock_doc.exists:

            lock_data = (
                current_lock_doc.to_dict()
                or {}
            )

            # Validate that the lock belongs
            # to this exact payment context.

            locked_payment_id = (
                lock_data.get(
                    "paymentId"
                )
            )

            locked_listing_id = (
                lock_data.get(
                    "listingId"
                )
            )

            locked_buyer_id = (
                lock_data.get(
                    "buyerId"
                )
            )

            if (
                locked_payment_id
                and
                locked_payment_id
                != payment_id
            ):
                raise Exception(
                    "Payment lock payment mismatch"
                )

            if (
                locked_listing_id
                and
                locked_listing_id
                != listing_id
            ):
                raise Exception(
                    "Payment lock listing mismatch"
                )

            if (
                locked_buyer_id
                and
                locked_buyer_id
                != buyer_id
            ):
                raise Exception(
                    "Payment lock buyer mismatch"
                )

            if (
                lock_data.get(
                    "status"
                )
                == "finalized"
                and
                lock_data.get(
                    "commerceOrderId"
                )
            ):
                return {
                    "alreadyProcessed":
                        True,

                    "orderId":
                        lock_data[
                            "commerceOrderId"
                        ],
                }

        # ======================================
        # CHECK LISTING
        # ======================================

        if not current_listing_doc.exists:
            raise Exception(
                "Listing not found"
            )

        current_listing = (
            current_listing_doc.to_dict()
            or {}
        )

        if current_listing.get(
            "sold",
            False,
        ):
            raise Exception(
                "Listing already sold"
            )

        if (
            current_listing.get(
                "status"
            )
            != "active"
        ):
            raise Exception(
                "Listing is no longer active"
            )

        if not current_listing.get(
            "instantBuyEnabled",
            False,
        ):
            raise Exception(
                "Instant Buy is no longer available"
            )

        # ======================================
        # RECHECK PRICE INSIDE TRANSACTION
        # ======================================

        try:
            current_price = float(
                current_listing.get(
                    "instantBuyPrice",
                    0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            raise Exception(
                "Invalid Instant Buy price"
            )

        current_expected_amount = int(
            round(
                current_price * 100
            )
        )

        if (
            current_expected_amount
            != expected_amount
        ):
            raise Exception(
                "Listing price changed"
            )

        # ======================================
        # CREATE COMMERCE ORDER
        # ======================================

        transaction.set(
            order_ref,
            {
                "orderId":
                    order_ref.id,

                "listingId":
                    listing_id,

                "buyerId":
                    buyer_id,

                "sellerId":
                    current_listing.get(
                        "sellerId"
                    ),

                "cropName":
                    current_listing.get(
                        "cropName"
                    ),

                "image":
                    current_listing.get(
                        "cropImage"
                    ),

                "quantity":
                    current_listing.get(
                        "quantity"
                    ),

                "unit":
                    current_listing.get(
                        "unit"
                    ),

                "amount":
                    current_price,

                "orderAmount":
                    current_price,

                "pickupOtp":
                    pickup_otp,

                "acceptedAmount":
                    None,

                "pickupDate":
                    None,

                "pickupTime":
                    "",

                "pickupLocation":
                    "",

                "paymentId":
                    payment_id,

                "razorpayOrderId":
                    razorpay_order_id,

                "paymentStatus":
                    "paid",

                "orderStatus":
                    "confirmed",

                "type":
                    "instant_buy",

                "pickupScheduled":
                    False,

                "otpVerified":
                    False,

                "sellerLatitude":
                    current_listing.get(
                        "latitude"
                    ),

                "sellerLongitude":
                    current_listing.get(
                        "longitude"
                    ),

                "sellerLocation":
                    current_listing.get(
                        "locationName"
                    ),

                "buyerNote":
                    "",

                "buyerNoteStatus":
                    "none",

                "pickupRequested":
                    False,

                "pickupRequestStatus":
                    "none",

                "pickupCharge":
                    0,

                "pickupMessage":
                    "",

                "pickupAddress":
                    "",

                "pickupAcceptedAt":
                    None,

                "webhookConfirmed":
                    False,

                "createdAt":
                    firestore.SERVER_TIMESTAMP,

                "updatedAt":
                    firestore.SERVER_TIMESTAMP,
            },
        )

        # ======================================
        # LOCK / MARK LISTING SOLD
        # ======================================

        transaction.update(
            listing_ref,
            {
                "sold":
                    True,

                "status":
                    "sold",

                "soldType":
                    "instant_buy",

                "soldTo":
                    buyer_id,

                "biddingEnabled":
                    False,

                "instantBuyEnabled":
                    False,

                "orderLocked":
                    True,

                "soldAt":
                    firestore.SERVER_TIMESTAMP,
            },
        )

        # ======================================
        # CREATE PAYMENT IDEMPOTENCY LOCK
        # ======================================

        transaction.set(
            payment_lock_ref,
            {
                "razorpayOrderId":
                    razorpay_order_id,

                "paymentId":
                    payment_id,

                "listingId":
                    listing_id,

                "buyerId":
                    buyer_id,

                "commerceOrderId":
                    order_ref.id,

                "status":
                    "finalized",

                "createdAt":
                    firestore.SERVER_TIMESTAMP,

                "updatedAt":
                    firestore.SERVER_TIMESTAMP,
            },
        )

        return {
            "alreadyProcessed":
                False,

            "orderId":
                order_ref.id,
        }

    # ==========================================
    # 9. EXECUTE TRANSACTION
    # ==========================================

    result = (
        finalize_transaction(
            transaction
        )
    )

    # ==========================================
    # 10. UPDATE ACTIVE BIDS AFTER TRANSACTION
    # ==========================================

    if not result[
        "alreadyProcessed"
    ]:

        active_bids = (
            db.collection(
                "commerce_bids"
            )
            .where(
                "listingId",
                "==",
                listing_id,
            )
            .where(
                "status",
                "==",
                "active",
            )
            .stream()
        )

        bid_batch = (
            db.batch()
        )

        has_bids = False

        for bid in active_bids:

            has_bids = True

            bid_batch.update(
                bid.reference,
                {
                    "status":
                        "outbid",

                    "updatedAt":
                        firestore.SERVER_TIMESTAMP,
                },
            )

        if has_bids:
            bid_batch.commit()

    # ==========================================
    # 11. RETURN FINAL RESULT
    # ==========================================

    return {
        "success":
            True,

        "alreadyProcessed":
            result[
                "alreadyProcessed"
            ],

        "orderId":
            result[
                "orderId"
            ],

        "orderStatus":
            "confirmed",
    }
def finalize_instant_buy_with_retry(
    razorpay_order_id: str,
    payment_id: str,
    listing_id: str,
    buyer_id: str,
    max_attempts: int = 3,
):
    
    import time
    from google.api_core.exceptions import Aborted

    for attempt in range(1, max_attempts + 1):

        try:
            return finalize_instant_buy(
                razorpay_order_id,
                payment_id,
                listing_id,
                buyer_id,
            )

        except Aborted:

            if attempt >= max_attempts:
                raise

            # Small backoff before retrying.
            time.sleep(
                0.25 * attempt
            )
def finalize_auction_payment_with_retry(
    razorpay_order_id: str,
    payment_id: str,
    order_id: str,
    buyer_id: str,
    max_attempts: int = 3,
):
    import time
    from google.api_core.exceptions import Aborted

    for attempt in range(1, max_attempts + 1):
        try:
            return finalize_auction_payment(
                razorpay_order_id,
                payment_id,
                order_id,
                buyer_id,
            )

        except Aborted:

            if attempt >= max_attempts:
                raise

            time.sleep(0.25 * attempt)

def finalize_cart(
    razorpay_order_id: str,
    payment_id: str,
    checkout_id: str,
    buyer_id: str,
):


    checkout_ref = (
        db.collection("commerce_checkouts")
        .document(checkout_id)
    )

    checkout_doc = checkout_ref.get()

    if not checkout_doc.exists:
        raise Exception("Checkout not found")

    checkout = checkout_doc.to_dict()

    if checkout["buyerId"] != buyer_id:
        raise Exception("Unauthorized checkout")

    if checkout.get("paymentStatus") == "paid":
        return {
            "alreadyProcessed": True,
            "orderId": checkout.get("orderId"),
            "orderStatus": "confirmed",
        }

    

    transaction = db.transaction()

    @firestore.transactional
    def complete_checkout(transaction):
        subtotal = 0.0

        seller_totals = {}

        order_items = []
        order_ref = (
            db.collection("commerce_orders")
            .document()
        )

        order_id = order_ref.id

        for item in checkout["items"]:

            if item["productType"] == "farm":

                listing_ref = (
                    db.collection("commerce_listings")
                    .document(item["listingId"])
                )

            else:

                listing_ref = (
                    db.collection("commerce_shop_products")
                    .document(item["listingId"])
                )
       

            listing_doc = listing_ref.get(
                transaction=transaction,
            )


            if not listing_doc.exists:

                raise Exception(
                    f"Product not found: {item['listingId']}"
                )

            listing = listing_doc.to_dict()

            quantity = float(
                item["quantity"]
            )

            if item["productType"] == "farm":

                available = float(
                    listing.get("quantity", 0)
                )

            else:

                available = float(
                    listing.get("stock", 0)
                )

            if quantity > available:

                raise Exception(
                    f"{listing['cropName']} is out of stock"
                )
            if item["productType"] == "farm":

                transaction.update(
                    listing_ref,
                    {
                        "quantity": available - quantity,
                    },
                )

            else:

                transaction.update(
                    listing_ref,
                    {
                        "stock": available - quantity,
                    },
                )
        
            line_total = quantity * float(
                listing["pricePerUnit"]
            )

            subtotal += line_total

            seller_id = listing["sellerId"]

            seller_totals[seller_id] = (

                seller_totals.get(
                    seller_id,
                    0,
                )

                + line_total

            )
            order_item_ref = (
                db.collection("commerce_order_items")
                .document()
            )

            transaction.set(

                order_item_ref,

                {

                    "orderItemId": order_item_ref.id,

                    "orderId": order_id,

                    "checkoutId": checkout_id,

                    "buyerId": buyer_id,

                    "sellerId": seller_id,

                    "listingId": item["listingId"],

                    "productName": item["productName"],

                    "productImage": (
                        item.get("image")
                        or listing.get("cropImage")
                        or listing.get("image")
                    ),

                    "quantity": quantity,

                    "unit": item["unit"],

                    "pricePerUnit": item["pricePerUnit"],

                    "subtotal": line_total,

                    "paymentStatus": "paid",

                    "orderStatus": "confirmed",

                    "paymentId": payment_id,

                    "createdAt": firestore.SERVER_TIMESTAMP,

                },

            )
        transaction.set(

            order_ref,

            {

                "orderId": order_id,

                "checkoutId": checkout_id,

                "buyerId": buyer_id,

                "paymentStatus": "paid",

                "orderStatus": "confirmed",

                "subtotal": subtotal,

                "deliveryCharge": checkout["deliveryCharge"],

                "grandTotal": checkout["grandTotal"],

                "paymentId": payment_id,

                "razorpayOrderId": razorpay_order_id,

                "createdAt": firestore.SERVER_TIMESTAMP,

                "settlementStatus": "pending_delivery",

                "settlementReleased": False,

            },

        )
        transaction.update(

            checkout_ref,

            {

                "paymentStatus": "paid",

                "orderStatus": "confirmed",

                "paymentId": payment_id,

                "razorpayOrderId": razorpay_order_id,

                "orderId": order_id,

                "paidAt": firestore.SERVER_TIMESTAMP,

            },

        )
        transaction.update(

            checkout_ref,

            {

                "settlementStatus": "pending_delivery",

                "settlementReleased": False,

            },

        )
        cart_query = (
            db.collection("commerce_cart")
            .where("buyerId", "==", buyer_id)
            .stream()
        )

        for cart_doc in cart_query:

            transaction.delete(
                cart_doc.reference,
            )
        return {
            "alreadyProcessed": False,
            "orderId": order_id,
            "orderStatus": "confirmed",
        }
    result = complete_checkout(transaction)

    return result


        
        
def finalize_cart_with_retry(
    razorpay_order_id: str,
    payment_id: str,
    checkout_id: str,
    buyer_id: str,
    max_attempts: int = 3,
):

    import time
    from google.api_core.exceptions import Aborted

    for attempt in range(1, max_attempts + 1):

        try:

            return finalize_cart(
                razorpay_order_id=razorpay_order_id,
                payment_id=payment_id,
                checkout_id=checkout_id,
                buyer_id=buyer_id,
            )

        except Aborted:

            if attempt >= max_attempts:
                raise

            time.sleep(0.25 * attempt)

import random

def finalize_auction_payment(
    razorpay_order_id: str,
    payment_id: str,
    order_id: str,
    buyer_id: str,
):

    # Step 3
    if not razorpay_order_id:
        raise Exception("Missing Razorpay order ID")

    if not payment_id:
        raise Exception("Missing payment ID")

    if not order_id:
        raise Exception("Missing order ID")

    if not buyer_id:
        raise Exception("Missing buyer ID")

    # Step 4
    order_ref = (
        db.collection("commerce_orders")
        .document(order_id)
    )

    # Step 5
    order_doc = order_ref.get()

    if not order_doc.exists:
        raise Exception("Order not found")

    order = order_doc.to_dict()

    # Step 6
    if order["buyerId"] != buyer_id:
        raise Exception("Buyer mismatch")

    # Step 7
    if order.get("paymentStatus") == "paid":
        return {
            "alreadyProcessed": True,
            "orderId": order_id,
            "paymentStatus": "paid",
            "orderStatus": order.get(
                "orderStatus",
                "confirmed",
            ),
        }

    # Step 8
    payment = razorpay_client.payment.fetch(
        payment_id
    )
    razorpay_order = razorpay_client.order.fetch(
    razorpay_order_id
    )

    notes = razorpay_order.get("notes", {})

    if notes.get("orderId") != order_id:
        raise Exception("Order mismatch")

    if notes.get("buyerId") != buyer_id:
        raise Exception("Buyer mismatch")

    # Step 9
    if payment["order_id"] != razorpay_order_id:
        raise Exception("Payment order mismatch")

    # Step 10
    if payment["status"] not in [
        "authorized",
        "captured",
    ]:
        raise Exception("Payment not successful")

    # Step 11
    order_amount = float(order["orderAmount"])

    # Step 12
    paid_amount = payment["amount"] / 100

    if abs(paid_amount - order_amount) > 0.01:
        raise Exception("Amount mismatch")
    expected_amount = int(round(order_amount * 100))

    if razorpay_order["amount"] != expected_amount:
        raise Exception("Razorpay order amount mismatch")

    seller_id = order["sellerId"]

    seller_ref = (
        db.collection("commerce_users")
        .document(seller_id)
    )

    wallet_tx_ref = (
        db.collection("wallet_transactions")
        .document()
    )

    payment_lock_ref = (
        db.collection("commerce_payment_locks")
        .document(razorpay_order_id)
    )

    pickup_otp = f"{random.randint(0, 999999):06d}"

    transaction = db.transaction()

    @firestore.transactional
    def update_paid_order(transaction):

        lock_doc = transaction.get(payment_lock_ref)

        if hasattr(lock_doc, "__next__"):
            lock_doc = next(lock_doc)

        fresh_order = transaction.get(order_ref)

        if hasattr(fresh_order, "__next__"):
            fresh_order = next(fresh_order)

        if not fresh_order.exists:
            raise Exception("Order disappeared")

        fresh = fresh_order.to_dict()
        existing_payment_id = fresh.get("paymentId")

        if (
            existing_payment_id
            and existing_payment_id != payment_id
        ):
            raise Exception(
                "Payment ID mismatch"
            )

        existing_razorpay_order = fresh.get(
            "razorpayOrderId"
        )

        if (
            existing_razorpay_order
            and existing_razorpay_order != razorpay_order_id
        ):
            raise Exception(
                "Razorpay Order ID mismatch"
            )

# Check payment lock first
        if lock_doc.exists:

            lock = lock_doc.to_dict() or {}

            if lock.get("status") == "finalized":

                return {
                    "alreadyProcessed": True,
                    "orderId": order_id,
                    "paymentStatus": "paid",
                    "orderStatus": fresh.get(
                    "orderStatus",
                    "confirmed",
                    ),
                }

# Fallback check
        if fresh.get("paymentStatus") == "paid":

            return {
                "alreadyProcessed": True,
                "orderId": order_id,
                "paymentStatus": "paid",
                "orderStatus": fresh.get(
                    "orderStatus",
                    "confirmed",
                ),
            }

        transaction.update(
            order_ref,
            {
                "paymentStatus": "paid",
                "orderStatus": "confirmed",
                "paymentId": payment_id,
                "razorpayOrderId": razorpay_order_id,
                "pickupOtp": pickup_otp,
                "paidAt": firestore.SERVER_TIMESTAMP,
                "updatedAt": firestore.SERVER_TIMESTAMP,
                "webhookConfirmed": True,
            },
        )
        transaction.set(
            payment_lock_ref,
            {
                "paymentId": payment_id,
                "buyerId": buyer_id,
                "orderId": order_id,
                "razorpayOrderId": razorpay_order_id,
                "status": "finalized",
                "createdAt": firestore.SERVER_TIMESTAMP,
                "updatedAt": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

        return {
            "alreadyProcessed": False,
            "orderId": order_id,
            "paymentStatus": "paid",
            "orderStatus": "confirmed",
        }
    result = update_paid_order(transaction)

    print(result)

    updated_order = order_ref.get()

    print(updated_order.to_dict())
    return result

@app.post("/payments/razorpay/verify")
async def verify_razorpay_payment(
    request: VerifyRazorpayPaymentRequest,
    user=Depends(verify_firebase_token),
):
    try:
        # ==========================================
        # 1. GET AUTHENTICATED BUYER
        # ==========================================

        buyer_id = user["uid"]

        # ==========================================
        # 2. VALIDATE REQUIRED PAYMENT DATA
        # ==========================================

        if not request.razorpay_order_id:
            raise HTTPException(
                status_code=400,
                detail="Missing Razorpay order ID",
            )

        if not request.razorpay_payment_id:
            raise HTTPException(
                status_code=400,
                detail="Missing Razorpay payment ID",
            )

        if not request.razorpay_signature:
            raise HTTPException(
                status_code=400,
                detail="Missing Razorpay signature",
            )

        

        # ==========================================
        # 3. VERIFY RAZORPAY CHECKOUT SIGNATURE
        # ==========================================

        try:
            razorpay_client.utility.verify_payment_signature(
                {
                    "razorpay_order_id":
                        request.razorpay_order_id,

                    "razorpay_payment_id":
                        request.razorpay_payment_id,

                    "razorpay_signature":
                        request.razorpay_signature,
                }
            )

        except Exception as signature_error:

            print(
                "RAZORPAY SIGNATURE VERIFICATION FAILED:",
                str(signature_error),
            )

            raise HTTPException(
                status_code=400,
                detail="Payment verification failed",
            )

        # ==========================================
        # 4. CALL SHARED PAYMENT FINALIZATION
        # ==========================================
        #
        # finalize_instant_buy() handles:
        #
        # - Razorpay payment verification
        # - Razorpay order verification
        # - Buyer ID verification
        # - Listing ID verification
        # - Payment amount verification
        # - Existing order/idempotency check
        # - Firestore transaction
        # - Order creation
        # - Listing sold/locked state
        # - Active bid updates
        #
        # Both Flutter /verify and Razorpay webhook
        # now use the same finalization function.
        # ==========================================
        if (
            request.checkoutId is None
            and request.orderId is None
            and request.listingId is None
        ):
            raise HTTPException(
                status_code=400,
                detail="checkoutId, orderId or listingId required",
            )
        print("========== VERIFY REQUEST ==========")
        print("checkoutId:", request.checkoutId)
        print("paymentId:", request.razorpay_payment_id)
        print("orderId:", request.razorpay_order_id)
        print("====================================")

        if request.checkoutId:
            print("ENTERED CART CHECKOUT FLOW")

            result = await asyncio.to_thread(
                finalize_cart_with_retry,
                request.razorpay_order_id,
                request.razorpay_payment_id,
                request.checkoutId,
                buyer_id,
                
            )

        elif request.orderId:

            result = await asyncio.to_thread(
                finalize_auction_payment_with_retry,
                request.razorpay_order_id,
                request.razorpay_payment_id,
                request.orderId,
                buyer_id,
            )

        else:

            result = await asyncio.to_thread(
                finalize_instant_buy_with_retry,
                request.razorpay_order_id,
                request.razorpay_payment_id,
                request.listingId,
                buyer_id,
            )

        # ==========================================
        # 5. RETURN FINAL ORDER RESULT
        # ==========================================

        return {
            "success":
                True,

            "alreadyProcessed":
                result.get(
                    "alreadyProcessed",
                    False,
                ),

            "orderId":
                result.get(
                    "orderId"
                ),

            "orderStatus":
                result.get(
                    "orderStatus",
                    "confirmed",
                ),
        }

    except HTTPException:
        raise

    

    except Exception as e:

        error_message = str(e)

        print("========== VERIFY ERROR ==========")

        traceback.print_exc()

        print("==================================")

        print(
            "RAZORPAY VERIFY ERROR:",
            error_message,
        )

                # ==========================================
        # LISTING UNAVAILABLE AFTER PAYMENT
        # ==========================================

        if (
            "already sold"
            in error_message.lower()
            or
            "no longer active"
            in error_message.lower()
            or
            "instant buy is no longer available"
            in error_message.lower()
        ):

            try:
                # ----------------------------------
                # INITIATE AUTOMATIC REFUND
                # ----------------------------------

                refund_result = await asyncio.to_thread(
                    refund_failed_instant_buy,
                    request.razorpay_payment_id,
                    request.razorpay_order_id,
                    request.listingId,
                    buyer_id,
                    "Listing became unavailable after payment",
                )

                print(
                    "RAZORPAY REFUND INITIATED:",
                    request.razorpay_payment_id,
                    refund_result.get(
                        "refundId"
                    ),
                )

                raise HTTPException(
                    status_code=409,
                    detail={
                        "code":
                            "LISTING_UNAVAILABLE_REFUND_INITIATED",

                        "message":
                            (
                                "This listing was purchased "
                                "by another buyer. Your payment "
                                "refund has been initiated."
                            ),

                        "refundId":
                            refund_result.get(
                                "refundId"
                            ),

                        "refundStatus":
                            refund_result.get(
                                "status"
                            ),
                    },
                )

            except HTTPException:
                raise

            except Exception as refund_error:

                print(
                    "CRITICAL RAZORPAY REFUND ERROR:",
                    request.razorpay_payment_id,
                    str(refund_error),
                )

                # ----------------------------------
                # SAVE FOR MANUAL RECONCILIATION
                # ----------------------------------

                reconciliation_ref = (
                    db.collection(
                        "commerce_payment_reconciliation"
                    )
                    .document(
                        request.razorpay_payment_id
                    )
                )

                reconciliation_ref.set(
                    {
                        "paymentId":
                            request.razorpay_payment_id,

                        "razorpayOrderId":
                            request.razorpay_order_id,

                        "listingId":
                            request.listingId,

                        "buyerId":
                            buyer_id,

                        "reason":
                            (
                                "Listing unavailable "
                                "after successful payment"
                            ),

                        "refundError":
                            str(
                                refund_error
                            )[:500],

                        "status":
                            "manual_review_required",

                        "createdAt":
                            firestore.SERVER_TIMESTAMP,

                        "updatedAt":
                            firestore.SERVER_TIMESTAMP,
                    },
                    merge=True,
                )

                raise HTTPException(
                    status_code=500,
                    detail={
                        "code":
                            "REFUND_REQUIRES_REVIEW",

                        "message":
                            (
                                "Your payment was successful, "
                                "but the order could not be completed. "
                                "The payment has been flagged for "
                                "refund review."
                            ),
                    },
                )

        # ==========================================
        # PAYMENT / ORDER MISMATCH
        # ==========================================

        if (
            "payment order mismatch"
            in error_message.lower()
            or
            "payment amount mismatch"
            in error_message.lower()
            or
            "razorpay order amount mismatch"
            in error_message.lower()
            or
            "listing mismatch"
            in error_message.lower()
            or
            "buyer mismatch"
            in error_message.lower()
        ):
            raise HTTPException(
                status_code=400,
                detail=error_message,
            )

        # ==========================================
        # PAYMENT NOT SUCCESSFUL
        # ==========================================

        if (
            "payment is not successful"
            in error_message.lower()
        ):
            raise HTTPException(
                status_code=400,
                detail="Payment is not successful",
            )

        # ==========================================
        # LISTING NOT FOUND
        # ==========================================

        if (
            "listing not found"
            in error_message.lower()
        ):
            raise HTTPException(
                status_code=404,
                detail="Listing not found",
            )

        # ==========================================
        # UNKNOWN SERVER ERROR
        # ==========================================

        raise HTTPException(
            status_code=500,
            detail="Unable to finalize payment",
        )
def claim_razorpay_webhook_event(
    event_id: str,
    event_type: str,
    payment_id: str | None,
    razorpay_order_id: str | None,
):
    """
    Atomically claim a Razorpay webhook event.

    Returns:
      claimed=True  -> process the event
      claimed=False -> event already handled or being processed
    """

    event_ref = (
        db.collection(
            "commerce_webhook_events"
        )
        .document(
            event_id
        )
    )

    transaction = db.transaction()

    @firestore.transactional
    def claim_transaction(transaction):

        event_doc = transaction.get(event_ref)

        if hasattr(event_doc, "__next__"):
            event_doc = next(event_doc)

        if not event_doc.exists:

            transaction.set(
                event_ref,
                {
                    "eventId": event_id,
                    "eventType": event_type,
                    "paymentId": payment_id,
                    "razorpayOrderId": razorpay_order_id,
                    "status": "processing",
                    "attemptCount": 1,
                    "createdAt": firestore.SERVER_TIMESTAMP,
                    "processingStartedAt": firestore.SERVER_TIMESTAMP,
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                },
            )

            return {
                "claimed": True,
                "status": "processing",
                "attemptCount": 1,
            }

        data = event_doc.to_dict() or {}

        status = data.get(
            "status"
        )

        attempt_count = int(
            data.get(
                "attemptCount",
                0,
            )
            or 0
        )

        # ======================================
        # ALREADY COMPLETELY HANDLED
        # ======================================

        if status in [
            "processed",
            "ignored",
            "refund_initiated",
        ]:

            return {
                "claimed":
                    False,

                "status":
                    status,

                "attemptCount":
                    attempt_count,
            }

        # ======================================
        # CURRENTLY BEING PROCESSED
        # ======================================

        if status == "processing":

            processing_started_at = (
                data.get(
                    "processingStartedAt"
                )
            )

            stale = False

            if processing_started_at:

                try:
                    now = datetime.now(
                        timezone.utc
                    )

                    stale = (
                        now
                        - processing_started_at
                        > timedelta(
                            minutes=10
                        )
                    )

                except Exception:
                    stale = False

            # Another worker is probably
            # still processing this event.

            if not stale:

                return {
                    "claimed":
                        False,

                    "status":
                        "processing",

                    "attemptCount":
                        attempt_count,
                }

            # Processing lock is stale.
            # Allow this retry to reclaim it.

            new_attempt_count = (
                attempt_count + 1
            )

            transaction.update(
                event_ref,
                {
                    "status":
                        "processing",

                    "attemptCount":
                        new_attempt_count,

                    "processingStartedAt":
                        firestore.SERVER_TIMESTAMP,

                    "lastRetryAt":
                        firestore.SERVER_TIMESTAMP,

                    "staleLockRecovered":
                        True,

                    "updatedAt":
                        firestore.SERVER_TIMESTAMP,
                },
            )

            return {
                "claimed":
                    True,

                "status":
                    "processing",

                "attemptCount":
                    new_attempt_count,
            }

        # ======================================
        # PREVIOUS ATTEMPT FAILED
        # ======================================
        #
        # processing_failed / refund_failed
        # can be claimed by a Razorpay retry.
        # ======================================

        new_attempt_count = (
            attempt_count + 1
        )

        transaction.update(
            event_ref,
            {
                "status":
                    "processing",

                "attemptCount":
                    new_attempt_count,

                "processingStartedAt":
                    firestore.SERVER_TIMESTAMP,

                "lastRetryAt":
                    firestore.SERVER_TIMESTAMP,

                "updatedAt":
                    firestore.SERVER_TIMESTAMP,
            },
        )

        return {
            "claimed":
                True,

            "status":
                "processing",

            "attemptCount":
                new_attempt_count,
        }

    result = claim_transaction(
        transaction
    )

    result["eventRef"] = (
        event_ref
    )

    return result
    
@app.post("/payments/razorpay/webhook")
async def razorpay_webhook(
    request: Request,
    x_razorpay_signature: str = Header(
        None,
        alias="X-Razorpay-Signature",
    ),
    x_razorpay_event_id: str = Header(
        None,
        alias="X-Razorpay-Event-Id",
    ),
):
    # --------------------------------
    # 1. READ RAW BODY
    # --------------------------------

    raw_body = await request.body()

    if not x_razorpay_signature:
        raise HTTPException(
            status_code=400,
            detail="Missing Razorpay signature",
        )

    # --------------------------------
    # 2. VERIFY WEBHOOK SIGNATURE
    # --------------------------------

    expected_signature = hmac.new(
        RAZORPAY_WEBHOOK_SECRET.encode(
            "utf-8"
        ),
        raw_body,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(
        expected_signature,
        x_razorpay_signature,
    ):
        print(
            "RAZORPAY WEBHOOK: "
            "INVALID SIGNATURE"
        )

        raise HTTPException(
            status_code=400,
            detail="Invalid webhook signature",
        )

    # --------------------------------
    # 3. PARSE ONLY AFTER VERIFICATION
    # --------------------------------

    try:
        payload = json.loads(
            raw_body.decode("utf-8")
        )

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid webhook payload",
        )

    event_type = payload.get(
        "event",
        "",
    )

    # --------------------------------
    # 4. REQUIRE EVENT ID
    # --------------------------------

    if not x_razorpay_event_id:
        raise HTTPException(
            status_code=400,
            detail="Missing Razorpay event ID",
        )
    
        # --------------------------------
    # 5. INITIALIZE WEBHOOK VARIABLES
    # --------------------------------

    payment_id = None
    razorpay_order_id = None
    listing_id = None
    buyer_id = None

        # --------------------------------
    # 6. EXTRACT PAYMENT ENTITY
    # --------------------------------

    payment_entity = (
        payload
        .get("payload", {})
        .get("payment", {})
        .get("entity", {})
    )

    if payment_entity:
        payment_id = payment_entity.get(
            "id"
        )

        razorpay_order_id = payment_entity.get(
            "order_id"
        )

        # ==========================================
    # ATOMICALLY CLAIM WEBHOOK EVENT
    # ==========================================

    claim_result = await asyncio.to_thread(
        claim_razorpay_webhook_event,
        x_razorpay_event_id,
        event_type,
        payment_id,
        razorpay_order_id,
    )

    event_ref = claim_result[
        "eventRef"
    ]

    if not claim_result.get(
        "claimed",
        False,
    ):

        existing_status = (
            claim_result.get(
                "status"
            )
        )

        print(
            "RAZORPAY WEBHOOK NOT CLAIMED:",
            x_razorpay_event_id,
            "status=",
            existing_status,
        )

        return {
            "success":
                True,

            "duplicate":
                True,

            "status":
                existing_status,
        }

    print(
        "RAZORPAY WEBHOOK CLAIMED:",
        x_razorpay_event_id,
        "attempt=",
        claim_result.get(
            "attemptCount"
        ),
    )
        



    try:

        # ==============================
        # PAYMENT CAPTURED / ORDER PAID
        # ==============================

        if event_type in [
            "payment.captured",
            "order.paid",
        ]:

            # --------------------------
            # VALIDATE PAYMENT DATA
            # --------------------------

            if not razorpay_order_id:

                event_ref.update({
                    "status":
                        "ignored",

                    "reason":
                        "Missing Razorpay order ID",

                    "processedAt":
                        firestore.SERVER_TIMESTAMP,
                })

                return {
                    "success": True,
                    "ignored": True,
                }

            if not payment_id:

                event_ref.update({
                    "status":
                        "ignored",

                    "reason":
                        "Missing Razorpay payment ID",

                    "processedAt":
                        firestore.SERVER_TIMESTAMP,
                })

                return {
                    "success": True,
                    "ignored": True,
                }

            # --------------------------
            # FETCH RAZORPAY ORDER
            # --------------------------

            razorpay_order = (
                razorpay_client
                .order
                .fetch(
                    razorpay_order_id
                )
            )

            notes = razorpay_order.get("notes") or {}

            buyer_id = notes.get("buyerId")
            listing_id = notes.get("listingId")
            order_id = notes.get("orderId")
            payment_type = notes.get("type", "instant_buy")

            # --------------------------
            # VALIDATE KISHANSEVA DATA
            # --------------------------

            if not buyer_id:

                event_ref.update({
                    "status": "ignored",
                    "reason": "Missing buyerId",
                    "processedAt": firestore.SERVER_TIMESTAMP,
                })

                return {
                    "success": True,
                    "ignored": True,
                }

            if payment_type == "auction":

                if not order_id:

                    event_ref.update({
                        "status": "ignored",
                        "reason": "Missing orderId",
                        "processedAt": firestore.SERVER_TIMESTAMP,
                    })

                    return {
                        "success": True,
                        "ignored": True,
                    }

            else:

                if not listing_id:

                    event_ref.update({
                        "status": "ignored",
                        "reason": "Missing listingId",
                        "processedAt": firestore.SERVER_TIMESTAMP,
                    })

                    return {
                        "success": True,
                        "ignored": True,
                    }

            # ======================================
            # SHARED INSTANT BUY FINALIZATION
            # ======================================
            #
            # This is the SAME function used by
            # /payments/razorpay/verify.
            #
            # If /verify already created the order,
            # it returns the existing order.
            #
            # If webhook arrives first,
            # it creates the order and locks listing.
            # ======================================

            if payment_type == "auction":

                result = await asyncio.to_thread(
                    finalize_auction_payment_with_retry,
                    razorpay_order_id,
                    payment_id,
                    order_id,
                    buyer_id,
                )

            else:

                result = await asyncio.to_thread(
                    finalize_instant_buy_with_retry,
                    razorpay_order_id,
                    payment_id,
                    listing_id,
                    buyer_id,
                )

            order_id = result.get(
                "orderId"
            )

            if not order_id:
                raise Exception(
                    "Order finalization returned no order ID"
                )

            # ======================================
            # MARK ORDER AS WEBHOOK CONFIRMED
            # ======================================

            order_ref = (
                db.collection(
                    "commerce_orders"
                )
                .document(
                    order_id
                )
            )

            order_ref.update({

                "webhookEventId":
                    x_razorpay_event_id,

                "updatedAt":
                    firestore.SERVER_TIMESTAMP,
            })

            # ======================================
            # MARK WEBHOOK EVENT PROCESSED
            # ======================================

            event_ref.update({
                "status":
                    "processed",

                "orderId":
                    order_id,

                "listingId":
                    listing_id,

                "buyerId":
                    buyer_id,

                "paymentId":
                    payment_id,

                "razorpayOrderId":
                    razorpay_order_id,

                "alreadyProcessed":
                    result.get(
                        "alreadyProcessed",
                        False,
                    ),

                "processedAt":
                    firestore.SERVER_TIMESTAMP,
            })

            print(
                "RAZORPAY WEBHOOK PROCESSED:",
                "event=",
                x_razorpay_event_id,
                "order=",
                order_id,
                "payment=",
                payment_id,
                "alreadyProcessed=",
                result.get(
                    "alreadyProcessed",
                    False,
                ),
            )

            return {
                "success":
                    True,

                "orderId":
                    order_id,

                "alreadyProcessed":
                    result.get(
                        "alreadyProcessed",
                        False,
                    ),
            }

        # ==============================
        # PAYMENT FAILED
        # ==============================

        elif event_type == "payment.failed":

            event_ref.update({
                "status":
                    "processed",

                "paymentStatus":
                    "failed",

                "paymentId":
                    payment_id,

                "razorpayOrderId":
                    razorpay_order_id,

                "processedAt":
                    firestore.SERVER_TIMESTAMP,
            })

            return {
                "success":
                    True,

                "paymentFailed":
                    True,
            }
                # ==============================
        # REFUND PROCESSED
        # ==============================

        elif event_type == "refund.processed":

            refund_entity = (
                payload
                .get("payload", {})
                .get("refund", {})
                .get("entity", {})
            )

            refund_id = refund_entity.get(
                "id"
            )

            refund_payment_id = (
                refund_entity.get(
                    "payment_id"
                )
            )

            if not refund_payment_id:

                event_ref.update({
                    "status":
                        "ignored",

                    "reason":
                        "Missing refund payment ID",

                    "processedAt":
                        firestore.SERVER_TIMESTAMP,
                })

                return {
                    "success": True,
                    "ignored": True,
                }

            refund_lock_ref = (
                db.collection(
                    "commerce_refund_locks"
                )
                .document(
                    refund_payment_id
                )
            )

            refund_lock_ref.set(
                {
                    "refundId":
                        refund_id,

                    "paymentId":
                        refund_payment_id,

                    "status":
                        "processed",

                    "processedAt":
                        firestore.SERVER_TIMESTAMP,

                    "updatedAt":
                        firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )

            event_ref.update({
                "status":
                    "processed",

                "refundId":
                    refund_id,

                "paymentId":
                    refund_payment_id,

                "refundStatus":
                    "processed",

                "processedAt":
                    firestore.SERVER_TIMESTAMP,
            })

            print(
                "RAZORPAY REFUND PROCESSED:",
                refund_id,
                refund_payment_id,
            )

            return {
                "success":
                    True,

                "refundProcessed":
                    True,

                "refundId":
                    refund_id,
            }

        # ==============================
        # OTHER EVENTS
        # ==============================

        else:

            event_ref.update({
                "status":
                    "ignored",

                "eventType":
                    event_type,

                "processedAt":
                    firestore.SERVER_TIMESTAMP,
            })

            return {
                "success":
                    True,

                "ignored":
                    True,
            }

    except Exception as e:

        error_message = str(e)

        print(
            "RAZORPAY WEBHOOK ERROR:",
            error_message,
        )

        # ==========================================
        # LISTING UNAVAILABLE AFTER PAYMENT
        # ==========================================

        if (
            "already sold"
            in error_message.lower()
            or
            "no longer active"
            in error_message.lower()
            or
            "instant buy is no longer available"
            in error_message.lower()
        ):

            try:
                # ----------------------------------
                # INITIATE IDEMPOTENT REFUND
                # ----------------------------------

                refund_result = await asyncio.to_thread(
                    refund_failed_instant_buy,
                    payment_id,
                    razorpay_order_id,
                    listing_id,
                    buyer_id,
                    "Listing became unavailable after payment",
                )

                # ----------------------------------
                # MARK WEBHOOK EVENT
                # ----------------------------------

                event_ref.update({
                    "status":
                        "refund_initiated",

                    "paymentId":
                        payment_id,

                    "razorpayOrderId":
                        razorpay_order_id,

                    "listingId":
                        listing_id,

                    "buyerId":
                        buyer_id,

                    "refundId":
                        refund_result.get(
                            "refundId"
                        ),

                    "refundStatus":
                        refund_result.get(
                            "status"
                        ),

                    "refundAlreadyRequested":
                        refund_result.get(
                            "alreadyRequested",
                            False,
                        ),

                    "reason":
                        "Listing unavailable after successful payment",

                    "processedAt":
                        firestore.SERVER_TIMESTAMP,
                })

                print(
                    "RAZORPAY WEBHOOK REFUND INITIATED:",
                    "payment=",
                    payment_id,
                    "refund=",
                    refund_result.get(
                        "refundId"
                    ),
                )

                # IMPORTANT:
                # Return HTTP 200.
                #
                # The payment has now been handled
                # by initiating a refund.
                #
                # Returning 500 here would cause
                # unnecessary webhook retries.

                return {
                    "success":
                        True,

                    "orderCreated":
                        False,

                    "refundInitiated":
                        True,

                    "refundId":
                        refund_result.get(
                            "refundId"
                        ),

                    "refundStatus":
                        refund_result.get(
                            "status"
                        ),
                }

            except Exception as refund_error:

                print(
                    "CRITICAL WEBHOOK REFUND ERROR:",
                    payment_id,
                    str(refund_error),
                )

                # ==================================
                # SAVE FOR MANUAL RECONCILIATION
                # ==================================

                if payment_id:

                    db.collection(
                        "commerce_payment_reconciliation"
                    ).document(
                        payment_id
                    ).set(
                        {
                            "paymentId":
                                payment_id,

                            "razorpayOrderId":
                                razorpay_order_id,

                            "listingId":
                                listing_id,

                            "buyerId":
                                buyer_id,

                            "reason":
                                (
                                    "Webhook detected unavailable "
                                    "listing after successful payment"
                                ),

                            "refundError":
                                str(
                                    refund_error
                                )[:500],

                            "status":
                                "manual_review_required",

                            "source":
                                "razorpay_webhook",

                            "webhookEventId":
                                x_razorpay_event_id,

                            "createdAt":
                                firestore.SERVER_TIMESTAMP,

                            "updatedAt":
                                firestore.SERVER_TIMESTAMP,
                        },
                        merge=True,
                    )

                event_ref.update({
                    "status":
                        "refund_failed",

                    "error":
                        str(
                            refund_error
                        )[:500],

                    "failedAt":
                        firestore.SERVER_TIMESTAMP,
                })

                # Return 500 because automated
                # handling has not succeeded.
                # This allows webhook retry.

                raise HTTPException(
                    status_code=500,
                    detail=
                        "Webhook refund processing failed",
                )

        # ==========================================
        # OTHER PROCESSING ERRORS
        # ==========================================

        event_ref.update({
            "status":
                "processing_failed",

            "error":
                error_message[:500],

            "failedAt":
                firestore.SERVER_TIMESTAMP,
        })

        raise HTTPException(
            status_code=500,
            detail="Webhook processing failed",
        )
@app.post("/orders/verify-pickup-otp")
async def verify_pickup_otp(
    request: VerifyPickupOtpRequest,
):
    if not request.orderId:
        raise HTTPException(
            status_code=400,
            detail="Order ID required",
        )

    if not request.otp:
        raise HTTPException(
            status_code=400,
            detail="OTP required",
        )
    order_ref = (
        db.collection("commerce_orders")
        .document(request.orderId)
    )

    order_doc = order_ref.get()

    if not order_doc.exists:
        raise HTTPException(
            status_code=404,
            detail="Order not found",
        )

    order = order_doc.to_dict()
    if order.get("paymentStatus") != "paid":
        raise HTTPException(
            status_code=400,
            detail="Payment not completed",
        )

    if order.get("orderStatus") != "pickup_scheduled":
        raise HTTPException(
            status_code=400,
            detail="Pickup not scheduled",
        )

    if order.get("otpVerified"):
        raise HTTPException(
            status_code=400,
            detail="Pickup already verified",
        )
    stored_otp = str(
        order.get("pickupOtp", "")
    ).strip()

    entered_otp = request.otp.strip()

    if stored_otp != entered_otp:
        raise HTTPException(
            status_code=400,
            detail="Invalid Pickup OTP",
        )
    

    transaction = db.transaction()
    @firestore.transactional
    def complete_order(transaction):
        fresh_order = transaction.get(order_ref)

        if hasattr(fresh_order, "__next__"):
            fresh_order = next(fresh_order)

        

       

        if not fresh_order.exists:
            raise Exception("Order not found")

       
        
        fresh = fresh_order.to_dict()
        order_items = list(

            db.collection("commerce_order_items")

            .where("orderId", "==", request.orderId)

            .stream()

        )

        if fresh.get("settlementReleased"):

            return {

                "success": True,

                "alreadyProcessed": True,

                "orderId": request.orderId,

                "orderStatus": "completed",

            }
        
        
       

        

       

        transaction.update(
            order_ref,
            {
                "otpVerified": True,
                "pickupOtpVerified": True,
                "pickupOtpVerifiedAt": firestore.SERVER_TIMESTAMP,
                "orderStatus": "completed",
                "completedAt": firestore.SERVER_TIMESTAMP,
                "updatedAt": firestore.SERVER_TIMESTAMP,
                "settlementReleased": True,
                "settlementStatus": "released",
                "settlementReleasedAt": firestore.SERVER_TIMESTAMP,
            },
        )
        for item_doc in order_items:

            item = item_doc.to_dict()

            seller_ref = (
                db.collection("commerce_users")
                .document(item["sellerId"])
            )

            seller_doc = transaction.get(seller_ref)

            if hasattr(seller_doc, "__next__"):
                seller_doc = next(seller_doc)

            if not seller_doc.exists:
                raise Exception("Seller not found")
            seller = seller_doc.to_dict()

            seller_amount = float(
                item["subtotal"]
            )

            transaction.update(

                seller_ref,

                {

                    "walletBalance": firestore.Increment(
                        seller_amount,
                    ),

                    "totalRevenue": firestore.Increment(
                        seller_amount,
                    ),

                    "totalSales": firestore.Increment(
                        1,
                    ),

                    "updatedAt": firestore.SERVER_TIMESTAMP,

                },

            )
            wallet_tx_ref = (
                db.collection("wallet_transactions")
                .document()
            )

            transaction.set(

                wallet_tx_ref,

                {

                    "transactionId": wallet_tx_ref.id,

                    "userId": item["sellerId"],

                    "orderId": request.orderId,

                    "buyerId": fresh["buyerId"],

                    "cropName": item["productName"],

                    "amount": seller_amount,

                    "commission": float(
                        item.get("platformCommission", 0),
                    ),

                    "type": "order_credit",

                    "status": "completed",

                    "createdAt": firestore.SERVER_TIMESTAMP,

                    "paymentId": fresh["paymentId"],

                },

            )
            transaction.update(

                item_doc.reference,

                {

                    "orderStatus": "completed",

                    "paymentStatus": "paid",

                    "settlementStatus": "released",

                    "settlementReleased": True,

                    "completedAt": firestore.SERVER_TIMESTAMP,

                    "updatedAt": firestore.SERVER_TIMESTAMP,

                },

            )

    
        

        

        
        
       
        return {
            "success": True,
            "orderId": request.orderId,
            "orderStatus": "completed",
        }
    result = complete_order(transaction)

    return result
# ================= NOTIFY ALL =================

@app.post("/notify-all")
def notify_all():

    users = db.collection("farmers").stream()

    sent = 0

    for user in users:

        data = user.to_dict()

        notifications_enabled = data.get(
            "notifications_enabled",
            True,
        )

        if not notifications_enabled:
            continue
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
                db.collection("farmers").document(user.id).update({
                    "fcm_token": None
                })
                print("🧹 Removed invalid token")
            else:
                print("❌ Error:", e)


# ================= TOPIC =================

@app.post("/notify-topic")
def notify_topic(
    title: str,
    body: str,
    lang: str = "en",
):

    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body,
        ),

        topic=f"alerts_{lang}",
    )

    messaging.send(message)

    return {
        "success": True,
        "topic": f"alerts_{lang}",
    }


# ================= DAILY =================

@app.get("/daily-reminder")
def daily_reminder():

    langs = ["en", "bn", "hi"]

    for lang in langs:

        if lang == "bn":

            title = "দৈনিক অনুস্মারক 🌱"
            body = "আজ আপনার ফসলের অবস্থা দেখুন"

        elif lang == "hi":

            title = "दैनिक अनुस्मारक 🌱"
            body = "आज अपनी फसल की स्थिति देखें"

        else:

            title = "Daily Reminder 🌱"
            body = "Check your crop health today"

        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body,
            ),

            topic=f"alerts_{lang}",
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

        return (weather, lang)

    except:
        return ""


def get_ndvi(lat, lon, user_id=None, field_id="default"):

    if user_id:
        try:
            doc = db.collection("fields")\
                .document(user_id)\
                .collection("user_fields")\
                .document(field_id)\
                .get()

            if doc.exists:
                data = doc.to_dict()
                if data.get("ndvi") is not None:
                    return data.get("ndvi")

        except Exception as e:
            print("NDVI fetch error:", e)

def get_users():

    users = db.collection("farmers").stream()

    data = []

    for u in users:

        d = u.to_dict()

        data.append({
            "id": u.id,
            "token": d.get("fcm_token"),
            "lat": d.get("lat"),
            "lon": d.get("lon"),
            "notifications_enabled": d.get(
                "notifications_enabled",
                True
            ),
            "field_id": d.get(
                "field_id",
                "default"
            )
        })

    return data
def get_scheme_alerts(user_id, lang="en"):

    alerts = []

    try:

        saved = db.collection("users")\
            .document(user_id)\
            .collection("savedSchemes")\
            .stream()

        now = datetime.utcnow()

        for s in saved:

            data = s.to_dict()

            if not data.get(
                "reminderEnabled",
                True
            ):
                continue

            scheme_doc = db.collection(
                "schemes"
            ).document(s.id).get()

            if not scheme_doc.exists:
                continue

            scheme = scheme_doc.to_dict()

            last_date = scheme.get(
                "lastDate"
            )

            if not last_date:
                continue

            diff = (
                last_date -
                now
            ).days

            if diff in [7, 3, 1]:

                title = (
                    "📢 Scheme Reminder"
                    if lang == "en"
                    else "📢 স্কিম রিমাইন্ডার"
                )

                body = (
                    f"{scheme.get('title')} closes in {diff} day(s)"
                    if lang == "en"
                    else f"{scheme.get('title')} এর শেষ তারিখ {diff} দিনের মধ্যে"
                )

                alerts.append(
                    (title, body)
                )

    except Exception as e:

        print(
            "Scheme alert error:",
            e
        )

    return alerts

def build_24_notifications(
    lang,
    weather,
    temp,
    humidity,
    ndvi,
    forecast,
    news_list,
    market_price,
):

    price = market_price

    alerts = []

    def t(en, bn, hi=None):

        if lang == "bn":
            return bn

        elif lang == "hi":
            return hi if hi else en

        return en

    # ================= 1. 🌧 RAIN TIMING =================
    for item in forecast.get("list", [])[:12]:

        cond = item.get(
            "weather",
            [{}]
        )[0].get(
            "main",
            ""
        ).lower()

        if "rain" in cond:

            ts = item.get("dt_txt")

            dt_utc = datetime.strptime(
                ts,
                "%Y-%m-%d %H:%M:%S"
            )

            dt_ist = dt_utc + timedelta(
                hours=5,
                minutes=30,
            )

            period = get_time_period(
                dt_ist,
                lang,
            )

            time_str = dt_ist.strftime(
                "%I:%M"
            )

            if lang == "bn":

                time_str = to_bengali_number(
                    time_str
                )

            elif lang == "hi":

                time_str = to_hindi_number(
                    time_str
                )

            today = datetime.utcnow() + timedelta(
                hours=5,
                minutes=30,
            )

            if dt_ist.date() == today.date():

                day_label = (
                    "আজ"
                    if lang == "bn"
                    else "आज"
                    if lang == "hi"
                    else "Today"
                )

            elif dt_ist.date() == (
                today + timedelta(days=1)
            ).date():

                day_label = (
                    "আগামীকাল"
                    if lang == "bn"
                    else "कल"
                    if lang == "hi"
                    else "Tomorrow"
                )

            else:

                day_label = (
                    to_bengali_number(
                        dt_ist.strftime(
                            "%d %b"
                        )
                    )

                    if lang == "bn"

                    else to_hindi_number(
                        dt_ist.strftime(
                            "%d %b"
                        )
                    )

                    if lang == "hi"

                    else dt_ist.strftime(
                        "%d %b"
                    )
                )

            alerts.append((

                t(
                    "🌧 Rain Alert",
                    "🌧 বৃষ্টি সতর্কতা",
                    "🌧 बारिश चेतावनी"
                ),

                t(
                    f"Rain expected {day_label} {period} around {time_str}",
                    f"{day_label} {period} {time_str} বৃষ্টি হওয়ার সম্ভাবনা আছে",
                    f"{day_label} {period} लगभग {time_str} बजे बारिश की संभावना है"
                )
            ))

            break

    # ================= 2. 🚨 NDVI / CROP =================
    if ndvi is not None:

        if ndvi < 0.3:

            alerts.append((

                t(
                    "🚨 Crop Critical",
                    "🚨 ফসল ঝুঁকিপূর্ণ",
                    "🚨 फसल संकट"
                ),

                t(
                    "Very low vegetation health",
                    "ফসলের অবস্থা খুব খারাপ",
                    "फसल की स्थिति बहुत खराब है"
                )
            ))

        elif ndvi < 0.5:

            alerts.append((

                t(
                    "⚠️ Crop Moderate",
                    "⚠️ মাঝারি ফসল",
                    "⚠️ मध्यम फसल"
                ),

                t(
                    "Moderate crop condition",
                    "ফসলের অবস্থা মাঝারি",
                    "फसल की स्थिति मध्यम है"
                )
            ))

        else:

            alerts.append((

                t(
                    "✅ Healthy Crop",
                    "✅ ভালো ফসল",
                    "✅ स्वस्थ फसल"
                ),

                t(
                    "Crop health is good",
                    "ফসলের অবস্থা ভালো",
                    "फसल की स्थिति अच्छी है"
                )
            ))

    # ================= 3. 💧 IRRIGATION =================
    rain_soon = False

    for item in forecast.get("list", [])[:12]:

        cond = item.get(
            "weather",
            [{}]
        )[0].get(
            "main",
            ""
        ).lower()

        if "rain" in cond:

            ts = item.get("dt_txt")

            dt_utc = datetime.strptime(
                ts,
                "%Y-%m-%d %H:%M:%S"
            )

            dt_ist = dt_utc + timedelta(
                hours=5,
                minutes=30,
            )

            now_ist = datetime.utcnow() + timedelta(
                hours=5,
                minutes=30,
            )

            if dt_ist <= now_ist + timedelta(hours=6):

                rain_soon = True

                break

    irrigation = get_irrigation(
        temp,
        humidity,
        ndvi,
    )

    if irrigation is not None:

        irrigation_val = round(
            float(irrigation),
            1,
        )

        if rain_soon:

            en_msg = (
                "Rain expected soon → skip irrigation"
            )

            bn_msg = (
                "শীঘ্রই বৃষ্টি হবে → সেচ বন্ধ রাখুন"
            )

            hi_msg = (
                "जल्द बारिश होगी → सिंचाई रोकें"
            )

        elif irrigation_val < 5:

            en_msg = (
                "No irrigation needed"
            )

            bn_msg = (
                "এখন সেচ প্রয়োজন নেই"
            )

            hi_msg = (
                "अभी सिंचाई की आवश्यकता नहीं है"
            )

        elif irrigation_val < 15:

            en_msg = (
                f"Light irrigation: {irrigation_val} mm"
            )

            bn_msg = (
                f"হালকা জলসেচ করুন: {irrigation_val} মিমি"
            )

            hi_msg = (
                f"हल्की सिंचाई करें: {irrigation_val} मिमी"
            )

        else:

            en_msg = (
                f"Apply irrigation: {irrigation_val} mm"
            )

            bn_msg = (
                f"{irrigation_val} মিমি জলসেচ করুন"
            )

            hi_msg = (
                f"{irrigation_val} मिमी सिंचाई करें"
            )

        if lang == "bn":

            irrigation_str = to_bengali_number(
                str(irrigation_val)
            )

            bn_msg = bn_msg.replace(
                str(irrigation_val),
                irrigation_str,
            )

        elif lang == "hi":

            irrigation_str = to_hindi_number(
                str(irrigation_val)
            )

            hi_msg = hi_msg.replace(
                str(irrigation_val),
                irrigation_str,
            )

        alerts.append((

            t(
                "💧 Irrigation Advice",
                "💧 সেচ পরামর্শ",
                "💧 सिंचाई सलाह"
            ),

            t(
                en_msg,
                bn_msg,
                hi_msg,
            )
        ))

    # ================= 4. 🌾 YIELD =================
    if ndvi is not None:

        yield_est = get_yield_prediction(
            ndvi,
            temp,
            60,
        )

        if yield_est is not None:

            y = round(
                float(yield_est),
                2,
            )

            if y < 2:

                en_msg = (
                    f"Low yield expected: {y} t/ha"
                )

                bn_msg = (
                    f"কম ফলনের সম্ভাবনা: {y} টন/হেক্টর"
                )

                hi_msg = (
                    f"कम उत्पादन की संभावना: {y} टन/हेक्टेयर"
                )

            elif y < 4:

                en_msg = (
                    f"Moderate yield expected: {y} t/ha"
                )

                bn_msg = (
                    f"মাঝারি ফলনের সম্ভাবনা: {y} টন/হেক্টর"
                )

                hi_msg = (
                    f"मध्यम उत्पादन की संभावना: {y} टन/हेक्टेयर"
                )

            else:

                en_msg = (
                    f"Good yield expected: {y} t/ha"
                )

                bn_msg = (
                    f"ভালো ফলনের সম্ভাবনা: {y} টন/হেক্টর"
                )

                hi_msg = (
                    f"अच्छे उत्पादन की संभावना: {y} टन/हेक्टेयर"
                )

        alerts.append((

            t(
                "🌾 Yield Forecast",
                "🌾 ফলন পূর্বাভাস",
                "🌾 उत्पादन पूर्वानुमान"
            ),

            t(
                en_msg,
                bn_msg,
                hi_msg,
            )
        ))

   

        temp_str = str(temp)

        if lang == "bn":

            temp_str = to_bengali_number(
                temp_str
            )
        elif lang == "hi":

            temp_str = to_hindi_number(
                temp_str
            )

        weather_msg = (
            f"{weather}, {temp_str}°C"
        )

        alerts.append((

            t(
                "🌤 Weather Update",
                "🌤 আবহাওয়ার আপডেট",
                "🌤 मौसम अपडेट"
            ),

            weather_msg
        ))
            


    alerts.append((

        t(
            "🌙 Tomorrow Planning",
            "🌙 আগামী দিনের পরিকল্পনা",
            "🌙 कल की योजना"
        ),

        t(
            "Prepare for tomorrow farming",
            "আগামীকালের কৃষিকাজের জন্য প্রস্তুতি নিন",
            "कल की खेती की तैयारी करें"
        )
    ))

    # ================= 6. 💰 MARKET =================

    
    
    if price:

        if lang == "bn":

            price = to_bengali_number(
                str(price)
            )

        elif lang == "hi":

            price = to_hindi_number(
                str(price)
            )

            alerts.append((

                t(
                    "💰 Market Price",
                    "💰 বাজার মূল্য",
                    "💰 बाजार मूल्य"
                ),

                price
            ))

        alerts.append((

            t(
                "📈 Sell Opportunity",
                "📈 বিক্রির সুযোগ",
                "📈 बिक्री का अवसर"
            ),

            t(
                "Prices may increase today",
                "আজ দাম বাড়তে পারে",
                "आज कीमत बढ़ सकती है"
            )
        ))

    # ================= 7. 📰 NEWS =================
    if news_list:

        for n in news_list[:2]:

            alerts.append((

                t(
                    "📰 Agri News",
                    "📰 কৃষি সংবাদ",
                    "📰 कृषि समाचार"
                ),

                n
            ))

    else:

        alerts.append((

            t(
                "📰 Agri News",
                "📰 কৃষি সংবাদ",
                "📰 कृषि समाचार"
            ),

            t(
                "Latest farming updates available",
                "নতুন কৃষি সংবাদ দেখুন",
                "नई कृषि जानकारी देखें"
            )
        ))

    # ================= 8. 🚨 STRESS =================
    if ndvi is not None and temp is not None:

        if ndvi < 0.4 and temp > 30:

            alerts.append((

                t(
                    "🚨 Crop Stress",
                    "🚨 ফসলের চাপ",
                    "🚨 फसल तनाव"
                ),

                t(
                    "Low NDVI + high temp → irrigate today",
                    "কম NDVI + বেশি তাপ → আজই জলসেচ করুন",
                    "कम NDVI + अधिक तापमान → आज सिंचाई करें"
                )
            ))

    # ================= 9. 📊 ENGAGEMENT =================
    alerts += [

        (
            t(
                "📊 NDVI Check",
                "📊 NDVI চেক করুন",
                "📊 NDVI जांचें"
            ),

            t(
                "See your crop health map",
                "আপনার জমির অবস্থা দেখুন",
                "अपनी फसल की स्थिति देखें"
            )
        ),

        (
            t(
                "📱 Open App",
                "📱 অ্যাপ খুলুন",
                "📱 ऐप खोलें"
            ),

            t(
                "Check farm insights now",
                "এখনই ফসলের তথ্য দেখুন",
                "अभी खेती की जानकारी देखें"
            )
        ),

        (
            t(
                "🧠 Smart Tip",
                "🧠 স্মার্ট পরামর্শ",
                "🧠 स्मार्ट सलाह"
            ),

            t(
                "AI can improve your yield",
                "AI ব্যবহার করে ফলন বাড়ান",
                "AI से उत्पादन बढ़ाएं"
            )
        ),
    ]

    # ================= 10. 🌱 TIPS =================
    tips = [

        (
            "Monitor crop daily",
            "নিয়মিত ফসল দেখুন",
            "फसल की नियमित निगरानी करें"
        ),

        (
            "Check soil moisture",
            "মাটির আর্দ্রতা পরীক্ষা করুন",
            "मिट्टी की नमी जांचें"
        ),

        (
            "Apply fertilizer if needed",
            "প্রয়োজনে সার দিন",
            "जरूरत होने पर उर्वरक दें"
        ),

        (
            "Check pest attack",
            "পোকামাকড় দেখুন",
            "कीट आक्रमण जांचें"
        ),

        (
            "Update farm diary",
            "ফার্ম ডায়েরি আপডেট করুন",
            "फार्म डायरी अपडेट करें"
        ),

        (
            "Use satellite insights",
            "স্যাটেলাইট ডাটা ব্যবহার করুন",
            "सैटेलाइट डेटा का उपयोग करें"
        ),
    ]

    for en, bn, hi in tips:

        alerts.append((

            t(
                "🌱 Tip",
                "🌱 পরামর্শ",
                "🌱 सलाह"
            ),

            t(
                en,
                bn,
                hi,
            )
        ))

    # ================= 11. FILL TO 24 =================
    i = 0

    while len(alerts) < 24:

        en, bn, hi = tips[
            i % len(tips)
        ]

        alerts.append((

            t(
                "🌱 Tip",
                "🌱 পরামর্শ",
                "🌱 सलाह"
            ),

            t(
                en,
                bn,
                hi,
            )
        ))

        i += 1

    return alerts[:24]

@app.get("/get-ndvi")
def get_ndvi_value(user_id: str, field_id: str):

    try:
        doc = db.collection("fields")\
            .document(user_id)\
            .collection("user_fields")\
            .document(field_id)\
            .get()

        if doc.exists:
            data = doc.to_dict()
            return {
                "ndvi": data.get("ndvi"),
                "updated": data.get("ndvi_updated")
            }

        return {"ndvi": None}

    except Exception as e:
        print("NDVI fetch error:", e)
        return {"ndvi": None}


    
    

@app.post("/smart-alerts")
def smart_alerts(data: dict):

    users = get_users()
    print("TOTAL USERS:", len(users))

    valid_tokens = sum(
        1 for u in users
        if u.get("token")
    )

    print("USERS WITH TOKENS:", valid_tokens)
    print(users)
    sent = 0
    print("TOTAL USERS:", len(users))
    #print("USER:", user_id, lat, lon, token)
    market_cache = get_market_price()
    
    for u in users:
        try:
            d = u
            print("USER DATA:", u)
            user_id = (
                u.get("id")
                or u.get("user_id")
                or u.get("uid")
            )
            notifications_enabled = d.get(
                "notifications_enabled",
                True,
            )

            if not user_id:
                continue

            if not notifications_enabled:
                continue

            if not notifications_enabled:
                continue
            #print("❌ Missing user_id → skip")
           # continue
            lang = get_user_lang(user_id)
            print(
                "USER:",
                user_id,
                "LANG:",
                lang,
            )
            
            news_cache = get_agri_news(lang)

            lat = u.get("lat")
            lon = u.get("lon")
            token = (
                u.get("token")
                or u.get("fcm_token")
                or u.get("device_token")
            
            )
            print(
                "TOKEN EXISTS:",
                bool(token)
            )
            # 🔥 HARD FILTER
            if not token:

                print("❌ Missing token")

                continue

            if lat is None or lon is None:

                print("⚠️ Missing lat/lon")

                weather = ""
                temp = 30
                humidity = 60
                forecast = {}
                ndvi = None

            else:

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
                field_id = u.get("field_id", "default")
                ndvi = get_ndvi(lat, lon, user_id, field_id)
            except:
                ndvi = None

            # ================= FORECAST =================
            forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={key}&units=metric"
            try:
                forecast = requests.get(forecast_url, timeout=3).json()
            except:
                forecast = {}

            # ================= ALERTS =================
            alerts = build_24_notifications(
                lang, weather, temp, humidity, ndvi, forecast, news_cache, market_cache
            )
            scheme_alerts = get_scheme_alerts(
                user_id,
                lang,
            )

            alerts.extend(
                scheme_alerts
            )

            # ================= COOLDOWN =================
            user_ref = db.collection("alerts_state").document(token)
            prev = user_ref.get().to_dict() or {}

            now = datetime.utcnow() + timedelta(hours=5, minutes=30)
            last_sent = prev.get("last_sent")

            if last_sent:
                last_time = datetime.fromisoformat(last_sent)
                if now - last_time < timedelta(minutes=55):
                    print("⛔ Cooldown active → skip user")
                    continue

            # ================= SEND =================
            if not alerts:
                continue

            # 12 AM – 4 AM window
            if hour < 4:
                continue

            prev_index = prev.get("last_index", -1)

            import random

# 🔥 ROTATION LOGIC
            index = (prev_index + 1) % len(alerts)

# 🔥 RANDOM BOOST (30%)
            if random.random() < 0.3:
                index = random.randint(0, len(alerts) - 1)

# 🔥 PRIORITY OVERRIDE
            priority = [

                a for a in alerts

                if (
                    "Rain" in a[0]
                    or "বৃষ্টি" in a[0]
                    or "बारिश" in a[0]
                )
            ]
            last_rain = prev.get("last_rain")
            last_rain_time = prev.get("last_rain_time")

            send_rain = False

# 🔥 check if new rain alert
            if priority:
                current_rain = priority[0][1]

                if current_rain != last_rain:
                    send_rain = True

# 🔥 cooldown (3 hours)
            if last_rain_time:
                try:
                    last_time = datetime.fromisoformat(last_rain_time)
                    if now - last_time < timedelta(hours=3):
                        send_rain = False
                except:
                    pass

# 🔥 FINAL DECISION
            if send_rain:
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
            print("✅ SENT TO:", user_id)
            sent += 1

# 🔥 SAVE STATE
            save_data = {
                "ndvi": ndvi,
                "last_sent": now.isoformat(),
                "last_index": index
}

            # 🔥 save rain memory
            if send_rain:
                save_data["last_rain"] = current_rain
                save_data["last_rain_time"] = now.isoformat()

            user_ref.set(save_data, merge=True)

            # ================= SAVE =================

        except Exception as e:
            print("FAILED USER:", user_id)
            print("TOKEN:", token)
            print("ERROR:", str(e))

            import traceback

            print("❌ Error:", e)

            traceback.print_exc()

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
