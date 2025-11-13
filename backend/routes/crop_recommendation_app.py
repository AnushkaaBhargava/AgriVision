from fastapi import APIRouter
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

router = APIRouter()

# --- Load trained crop recommendation model ---
with open("models/XGBoost-final-crop.pkl", "rb") as f:
    model = pickle.load(f)

# --- Load the label encoder used during training ---
encoder = pickle.load(open("models/label_encoder.pkl", "rb"))

@router.get("/recommend_crop")
async def recommend_crop(
    N: float, P: float, K: float,
    temperature: float, humidity: float,
    ph: float, rainfall: float
):
    # Prepare input features
    input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Get prediction (model returns encoded integer)
    prediction = model.predict(input_features)

    # Decode back to actual crop name using the encoder
    crop_name = encoder.inverse_transform(prediction)[0]

    return {"recommended_crop": crop_name}
