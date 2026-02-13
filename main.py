# -*- coding: utf-8 -*-
import os
import sys
import subprocess
from fastapi import FastAPI
from api import app  # 👈 IMPORT YOUR ROUTES

if os.getenv("RENDER"):
    try:
        print("[INFO] Downloading models...")
        subprocess.run([sys.executable, "download_models.py"], check=True)
    except Exception as e:
        print("[ERROR] Model download failed:", e)

@app.get("/")
def root():
    return {"status": "ok", "service": "KishanSeva AI"}
