from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Used Car Price Predictor - JSON API")

# Load model
model_path = os.path.join("models", "random_forest.pkl")
model = joblib.load(model_path)

# Request body schema
class CarFeatures(BaseModel):
    Year: int
    Present_Price: float
    Kms_Driven: int
    Fuel_Type: int
    Seller_Type: int
    Transmission: int
    Owner: int

@app.get("/")
def root():
    return {"message": "Car Price Prediction API is running!"}

@app.post("/predict")
def predict_price(features: CarFeatures):
    input_array = np.array([
        features.Year,
        features.Present_Price,
        features.Kms_Driven,
        features.Fuel_Type,
        features.Seller_Type,
        features.Transmission,
        features.Owner
    ]).reshape(1, -1)

    prediction = model.predict(input_array)[0]
    return {
        "predicted_selling_price": round(prediction, 2),
        "status": "success"
    }
