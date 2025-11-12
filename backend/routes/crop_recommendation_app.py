from fastapi import APIRouter, Query
import numpy as np
import pickle

router = APIRouter()

# ✅ Load trained model (replace .json with .pkl)
with open("models/XGBoost-final-crop.pkl", "rb") as f:
    model = pickle.load(f)

@router.get("/recommend_crop")
async def recommend_crop(
    N: int = Query(..., description="Nitrogen content in soil"),
    P: int = Query(..., description="Phosphorus content in soil"),
    K: int = Query(..., description="Potassium content in soil"),
    temperature: float = Query(..., description="Temperature in °C"),
    humidity: float = Query(..., description="Humidity percentage"),
    ph: float = Query(..., description="pH value of the soil"),
    rainfall: float = Query(..., description="Rainfall in mm")
):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)
    return {"recommended_crop": str(prediction[0])}
