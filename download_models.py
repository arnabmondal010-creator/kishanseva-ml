import os
import requests

BASE_URL = "https://huggingface.co/kishanseva-chatbot/kishanseva-plant-disease/resolve/main"

MODELS = {
    "plant_disease.h5": f"{BASE_URL}/plant_disease.h5",
    "rice_disease.h5": f"{BASE_URL}/rice_disease.h5",
}

os.makedirs("model", exist_ok=True)

for name, url in MODELS.items():
    out_path = os.path.join("model", name)
    if os.path.exists(out_path):
        print(f"[SKIP] {name} already exists")
        continue

    print(f"[INFO] Downloading {name}")
    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"[OK] Saved {name}")
