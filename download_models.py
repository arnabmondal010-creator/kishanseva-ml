Python 2.7.10 (default, May 23 2015, 09:40:32) [MSC v.1500 32 bit (Intel)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> import os
import requests

MODELS = {
    "https://huggingface.co/datasets/kishanseva-chatbot/kishanseva-plant-disease/blob/main/plant_disease.h5",
    "https://huggingface.co/datasets/kishanseva-chatbot/kishanseva-plant-disease/blob/main/rice_disease.h5",
}

os.makedirs("model", exist_ok=True)

for name, url in MODELS.items():
    path = os.path.join("model", name)
    if not os.path.exists(path):
        print(f"Downloading {name}...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    else:
        print(f"{name} already exists")
