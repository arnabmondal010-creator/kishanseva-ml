# api.py
import os
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODELS = {}
CLASS_NAMES = {
    "tomato": ["Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold", "Healthy"],
    "rice": ["Bacterial_leaf_blight", "Brown_spot", "Leaf_smut", "Healthy"]
}

MODEL_PATHS = {
    "tomato": "model/plant_disease.tflite",
    "rice": "model/rice_disease.tflite"
}

INPUT_SIZE = 224


def load_tflite_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter


@app.on_event("startup")
def load_models():
    print("[INFO] Loading TFLite models...")
    for crop, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            raise RuntimeError(f"Model file missing: {path}")
        MODELS[crop] = load_tflite_model(path)
    print("[OK] Models loaded")


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((INPUT_SIZE, INPUT_SIZE))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def run_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]
    return output


@app.get("/")
def root():
    return {"status": "KishanSeva ML API running"}


@app.post("/predict")
async def predict(
    crop: str = Form(...),
    file: UploadFile = File(...)
):
    crop = crop.lower()

    if crop not in MODELS:
        raise HTTPException(status_code=400, detail="Invalid crop type")

    image = Image.open(file.file)
    input_tensor = preprocess_image(image)

    probs = run_inference(MODELS[crop], input_tensor)
    idx = int(np.argmax(probs))
    confidence = float(np.max(probs))

    return {
        "crop": crop,
        "disease": CLASS_NAMES[crop][idx],
        "confidence": round(confidence, 4)
    }
