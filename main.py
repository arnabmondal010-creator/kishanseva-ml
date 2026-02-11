from fastapi import FastAPI, UploadFile, File, Form
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

import os
import subprocess

if not os.path.exists("model/plant_disease.h5") or not os.path.exists("model/rice_disease.h5"):
    subprocess.run(["python", "download_models.py"], check=True)


MODELS = {
    "tomato_potato_pepper": tf.keras.models.load_model("model/plant_disease.h5"),
    "rice": tf.keras.models.load_model("model/rice_disease.h5")
}

CLASS_NAMES = {
    "tomato_potato_pepper": [
        "Pepper__bell___Bacterial_spot",
        "Pepper__bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight",
        "Tomato_Late_blight",
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite",
        "Tomato__Target_Spot",
        "Tomato__Tomato_YellowLeaf__Curl_Virus",
        "Tomato__Tomato_mosaic_virus",
        "Tomato_healthy"
    ],
    "rice": [
        "Bacterial leaf blight",
        "Brown spot",
        "Leam smut",
    ]
}

@app.post("/predict")
async def predict(crop: str = Form(...), file: UploadFile = File(...)):
    model = MODELS[crop]
    labels = CLASS_NAMES[crop]

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((224,224))
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, 0)

    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))

    return {"crop": crop, "disease": labels[idx], "confidence": round(conf, 4)}
