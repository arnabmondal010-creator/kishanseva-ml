# -*- coding: utf-8 -*-
import razorpay
import os

client = razorpay.Client(
    auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET"))
)

PLANS = {
    "basic": 7900,   # ₹79 in paise
    "pro": 14900    # ₹149 in paise
}

def create_order(plan: str):
    amount = PLANS.get(plan)
    if not amount:
        raise ValueError("Invalid plan")

    order = client.order.create({
        "amount": amount,
        "currency": "INR",
        "payment_capture": 1
    })

    return order
