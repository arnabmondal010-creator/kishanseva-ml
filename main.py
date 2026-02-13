import os
import sys
import subprocess
from fastapi import FastAPI

app = FastAPI()

if os.getenv("RENDER"):
    try:
        print("[INFO] Downloading models...")
        subprocess.run([sys.executable, "download_models.py"], check=True)
    except Exception as e:
        print("[ERROR] Model download failed:", e)
