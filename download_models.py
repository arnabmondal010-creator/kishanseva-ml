import os
import sys
import requests

MODELS = {
    "plant_disease.h5": "https://huggingface.co/datasets/kishanseva-chatbot/kishanseva-plant-disease/resolve/main/plant_disease.h5",
    "rice_disease.h5": "https://huggingface.co/datasets/kishanseva-chatbot/kishanseva-plant-disease/resolve/main/rice_disease.h5",
}

os.makedirs("model", exist_ok=True)

def download(name, url):
    path = os.path.join("model", name)
    if os.path.exists(path):
        print(f"[OK] {name} already exists.")
        return True

    print(f"[INFO] Downloading {name}")
    r = requests.get(url, stream=True, timeout=600)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    print(f"[OK] Saved {name}")
    return True

ok = True
for name, url in MODELS.items():
    ok = download(name, url) and ok

if not ok:
    sys.exit(1)

