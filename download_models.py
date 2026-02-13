import os
import requests

os.makedirs("model", exist_ok=True)

MODELS = {
    "plant_disease.h5": "https://huggingface.co/kishanseva-chatbot/kishanseva-plant-disease/resolve/main/plant_disease.h5",
    "rice_disease.h5": "https://huggingface.co/kishanseva-chatbot/kishanseva-plant-disease/resolve/main/rice_disease.h5",
}

for name, url in MODELS.items():
    path = f"model/{name}"
    if not os.path.exists(path):
        print(f"[INFO] Downloading {name}")
        r = requests.get(url)
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"[OK] Saved {name}")
