# app.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load model
model = joblib.load("models/house_price_model.pkl")

@app.get("/")
def home():
    return {"message": "House Price Prediction API"}

@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        return {"predicted_price": float(prediction)}
    except Exception as e:
        return {"error": str(e)}