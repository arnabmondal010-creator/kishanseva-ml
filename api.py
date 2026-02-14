# -*- coding: utf-8 -*-
import os
import razorpay
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from limits import can_use
from ai_service import analyze_image

# -----------------------------
# App
# -----------------------------
app = FastAPI()

# -----------------------------
# Razorpay Setup (ENV VARS)
# -----------------------------
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")

if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
    raise RuntimeError("Razorpay keys not set in environment variables")

razorpay_client = razorpay.Client(
    auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)
)

PLANS = {
    "basic": 7900,   # ₹79
    "pro": 14900     # ₹149
}

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
    # TODO: Save user plan in DB
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

