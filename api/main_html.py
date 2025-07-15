from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
import os
from prometheus_fastapi_instrumentator import Instrumentator

# ✅ Create FastAPI app first
app = FastAPI()

# ✅ Set up Prometheus instrumentation after app is created
Instrumentator().instrument(app).expose(app)

# ✅ Jinja2 Templates
templates = Jinja2Templates(directory="api/templates")

# ✅ Load ML model
model = joblib.load("models/random_forest.pkl")  # or your best_model.pkl

# ✅ Home route (index form)
@app.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ✅ Predict route
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
    prediction = round(prediction, 2)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "prediction": prediction
    })
