from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
from PIL import Image
import io
import tensorflow as tf

app = FastAPI()

@app.get("/")
def root():
    return {"status": "KishanSeva AI API is running"}

@app.post("/predict")
async def predict(crop: str = Form(...), file: UploadFile = File(...)):
    return {
        "crop": crop,
        "disease": "test-disease",
        "confidence": 0.99
    }
