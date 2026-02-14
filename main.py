from fastapi import FastAPI
from api import app

app = FastAPI()

# Mount all API routes from api.py


@app.get("/")
def root():
    return {"status": "KishanSeva AI backend running"}
