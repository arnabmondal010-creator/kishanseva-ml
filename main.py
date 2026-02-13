import os
import sys
import subprocess
from api import app

if os.getenv("RENDER"):
    try:
        print("[INFO] Downloading models on startup...")
        subprocess.run([sys.executable, "download_models.py"], check=True)
    except Exception as e:
        print("[ERROR] Model download failed:", e)

@app.get("/")
def root():
    return {"status": "ok"}
