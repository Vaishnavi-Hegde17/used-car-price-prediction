from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
import os

app = FastAPI()
templates = Jinja2Templates(directory="api/templates")

# Load model
model = joblib.load("models/random_forest.pkl")  # or your best model path

@app.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request,
            Year: int = Form(...),
            Present_Price: float = Form(...),
            Kms_Driven: int = Form(...),
            Fuel_Type: int = Form(...),
            Seller_Type: int = Form(...),
            Transmission: int = Form(...),
            Owner: int = Form(...)):

    input_data = np.array([[Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner]])
    prediction = model.predict(input_data)[0]
    prediction = round(prediction, 2)  # e.g., 3.45 Lakhs

    return templates.TemplateResponse("result.html", {
        "request": request,
        "prediction": prediction
    })
