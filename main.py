import os
import sys
import subprocess
from fastapi import FastAPI

app = FastAPI()

if os.getenv("RENDER"):
    print("[INFO] Downloading models on startup...")
    result = subprocess.run([sys.executable, "download_models.py"])
    if result.returncode != 0:
        raise RuntimeError("Model download failed. Stopping server.")
