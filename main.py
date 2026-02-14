from fastapi import FastAPI
from api import app 

app = FastAPI(title="KishanSeva AI")

# Mount all API routes
app.mount("/", api_app)
