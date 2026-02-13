import os
import requests

MODELS = {
    "plant_disease.tflite": "https://huggingface.co/kishanseva-chatbot/kishanseva-tflite-models/resolve/main/plant_disease.tflite",
    "rice_disease.tflite": "https://huggingface.co/kishanseva-chatbot/kishanseva-tflite-models/resolve/main/rice_disease.tflite",
}

os.makedirs("model", exist_ok=True)

for name, url in MODELS.items():
    path = os.path.join("model", name)
    if not os.path.exists(path):
        print("[INFO] Downloading", name)
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print("[OK] Saved", name)
    else:
        print("[SKIP]", name, "already exists")
