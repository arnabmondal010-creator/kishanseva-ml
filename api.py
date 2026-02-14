# -*- coding: utf-8 -*-
import os
import razorpay
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from limits import can_use
from ai_service import analyze_image

app = FastAPI(title="KishanSeva AI API")

# -----------------------------
# Razorpay Setup (ENV VARS)
# -----------------------------
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")

if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
    print("⚠ Razorpay keys not set (payments will fail)")

razorpay_client = razorpay.Client(
    auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)
)

PLANS = {
    "basic": 7900,   # ₹79
    "pro": 14900     # ₹149
}

# -----------------------------
# Root
# -----------------------------
@app.get("/")
def root():
    return {"status": "KishanSeva AI running"}

# -----------------------------
# Create Payment Order
# -----------------------------
@app.post("/create-order")
def create_payment_order(plan: str = Form(...)):
    if plan not in PLANS:
        raise HTTPException(status_code=400, detail="Invalid plan")

    order = razorpay_client.order.create({
        "amount": PLANS[plan],
        "currency": "INR",
        "payment_capture": 1
    })

    return {
        "order_id": order["id"],
        "amount": order["amount"],
        "currency": "INR",
        "key": RAZORPAY_KEY_ID
    }

# -----------------------------
# Confirm Upgrade (TEMP)
# -----------------------------
@app.post("/confirm-upgrade")
def confirm_upgrade(plan: str = Form(...), user_id: str = Form(...)):
    return {
        "status": "success",
        "plan": plan,
        "user_id": user_id
    }

# -----------------------------
# Prediction API (OpenAI ONLY)
# -----------------------------
@app.post("/predict")
async def predict(
    user_id: str = Form(None),
    crop: str = Form(...),
    file: UploadFile = File(...)
):
    user_id = user_id or "guest_user"

    if not can_use(user_id):
        raise HTTPException(status_code=402, detail="Free limit exceeded. Please upgrade.")

    image_bytes = await file.read()
    result = await analyze_image(crop, image_bytes)

    return {
        "disease": result.get("disease", "Unknown disease"),
        "confidence": float(result.get("confidence", 0.0)),
        "advice": result.get(
            "advice",
            "No specific advice available. Please try a clearer photo or consult a local agriculture officer."
        )
    }


# -----------------------------
# Yield Prediction API
# -----------------------------
from joblib import load
import pandas as pd

yield_model = load("models/yield_model.joblib")

@app.post("/predict-yield")
async def predict_yield(
    soil_type: str = Form(...),
    fertilizer_type: str = Form(...),
    crop_stage: str = Form(...),
    stress_level: str = Form(...),
    fertilizer_kg: float = Form(...),
    irrigation_count: int = Form(...),
    pesticide_sprays: int = Form(...),
    avg_temp: float = Form(...),
    rainfall: float = Form(...),
    humidity: float = Form(...),
    wind_speed: float = Form(...),
    ndvi: float = Form(...)
):
    X = pd.DataFrame([{
        "soil_type": soil_type,
        "fertilizer_type": fertilizer_type,
        "crop_stage": crop_stage,
        "stress_level": stress_level,
        "fertilizer_kg": fertilizer_kg,
        "irrigation_count": irrigation_count,
        "pesticide_sprays": pesticide_sprays,
        "avg_temp": avg_temp,
        "rainfall": rainfall,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "ndvi": ndvi,
    }])

    y_pred = yield_model.predict(X)[0]

    return {
        "predicted_yield_kg_per_hectare": float(y_pred)
    }
