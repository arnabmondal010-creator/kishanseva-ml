# main.py
from api import app

# Optional root for health check
@app.get("/")
def root():
    return {"status": "KishanSeva AI running (Render)"}
