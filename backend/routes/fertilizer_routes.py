from fastapi import APIRouter
import pickle
import numpy as np
import pandas as pd

router = APIRouter()

# Load model
with open("models/fertilizer_recommendation.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
le = data["label_encoder"]

# Load CSV so we can fetch dosage, frequency, notes
df = pd.read_csv("data/Crop_recommendation_with_fertilizer_packages_full.csv")

@router.post("/predict")
def predict_fertilizer(N: float, P: float, K: float,
                       temperature: float, humidity: float,
                       ph: float, rainfall: float):

    # Prepare input
    inputs = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Model prediction
    encoded_pred = model.predict(inputs)[0]
    fert_package = le.inverse_transform([encoded_pred])[0]

    # Fetch the row with this fertilizer package
    row = df[df["Fertilizer_Package"] == fert_package].iloc[0]

    return {
        "fertilizer_package": fert_package,
        "dosage_kg_per_acre_or_tree": row["Dosage_kg_per_acre_or_tree"],
        "application_frequency": row["Application_Frequency"],
        "recommendation_notes": row["Recommendation_Notes"]
    }
