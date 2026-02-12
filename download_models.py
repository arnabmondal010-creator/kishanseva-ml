Python 2.7.10 (default, May 23 2015, 09:40:32) [MSC v.1500 32 bit (Intel)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> import os
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

    print(f"[INFO] Downloading {name} from {url}")
    try:
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        print(f"[OK] Saved {name} to {path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {name}: {e}", file=sys.stderr)
        return False

ok = True
for name, url in MODELS.items():
    ok = download(name, url) and ok

if not ok:
    sys.exit(1)
