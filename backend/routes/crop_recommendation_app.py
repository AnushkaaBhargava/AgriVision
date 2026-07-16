from fastapi import APIRouter, HTTPException
import pickle
import numpy as np
import json
import requests

router = APIRouter()

# ---------------------------------------------------------
# Load model + encoder + crop requirements
# ---------------------------------------------------------
MODEL_PATH = "models/XGBoost-final-crop.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
CROP_REQ_PATH = "data/crop_requirements.json"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

with open(CROP_REQ_PATH, "r") as f:
    crop_reqs = json.load(f)


# ---------------------------------------------------------
# Weather Fetch (RapidAPI)
# ---------------------------------------------------------
def fetch_weather(city: str):
    url = "https://open-weather13.p.rapidapi.com/city"
    headers = {
        "x-rapidapi-host": "open-weather13.p.rapidapi.com",
        "x-rapidapi-key": "26dae2fde4mshbb516e4e5795cd7p18b741jsnb2018c29bf8c"
    }
    params = {"city": city, "lang": "EN"}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)

        # If API fails → fallback default weather
        if resp.status_code != 200:
            return {
                "avg_temp_c": 26.0,
                "avg_humidity_pct": 60.0,
                "avg_wind_m_s": 2.0,
                "monthly_precip_mm_est": 150.0
            }

        data = resp.json()

        temp_raw = data["main"]["temp"]
        temp_c = (temp_raw - 32) * 5 / 9 if temp_raw > 60 else temp_raw

        humidity = data["main"].get("humidity", 60)
        wind_speed = data.get("wind", {}).get("speed", 2.0)

        return {
            "avg_temp_c": float(temp_c),
            "avg_humidity_pct": float(humidity),
            "avg_wind_m_s": float(wind_speed),
            "monthly_precip_mm_est": 150.0
        }

    except Exception:
        # Hard fallback for ANY error
        return {
            "avg_temp_c": 26.0,
            "avg_humidity_pct": 60.0,
            "avg_wind_m_s": 2.0,
            "monthly_precip_mm_est": 150.0
        }


# ---------------------------------------------------------
# Crop suitability score
# ---------------------------------------------------------
def score_crop(req, weather, soil):
    weights = req.get("weights", {"temp": 0.35, "rain": 0.35, "humidity": 0.15, "ph": 0.15})

    # Temperature
    if "temp_min" in req:
        t = weather["avg_temp_c"]
        if req["temp_min"] <= t <= req["temp_max"]:
            temp_score = 1.0
        else:
            dist = min(abs(t - req["temp_min"]), abs(t - req["temp_max"]))
            temp_score = max(0, 1 - dist / 10)
    else:
        temp_score = 0.6

    # Rainfall
    monthly = weather["monthly_precip_mm_est"]
    if "monthly_rain_min" in req:
        if req["monthly_rain_min"] <= monthly <= req["monthly_rain_max"]:
            rain_score = 1.0
        else:
            dist = min(abs(monthly - req["monthly_rain_min"]),
                       abs(monthly - req["monthly_rain_max"]))
            rain_score = max(0, 1 - dist / 100)
    else:
        rain_score = 0.6

    # Humidity
    if "humidity_min" in req:
        h = weather["avg_humidity_pct"]
        if req["humidity_min"] <= h <= req["humidity_max"]:
            hum_score = 1.0
        else:
            dist = min(abs(h - req["humidity_min"]),
                       abs(h - req["humidity_max"]))
            hum_score = max(0, 1 - dist / 50)
    else:
        hum_score = 0.6

    # Soil pH
    soil_ph = soil["ph"]
    if "ph_min" in req:
        if req["ph_min"] <= soil_ph <= req["ph_max"]:
            ph_score = 1.0
        else:
            dist = min(abs(soil_ph - req["ph_min"]),
                       abs(soil_ph - req["ph_max"]))
            ph_score = max(0, 1 - dist / 3)
    else:
        ph_score = 0.6

    final_score = (
        temp_score * weights["temp"] +
        rain_score * weights["rain"] +
        hum_score * weights["humidity"] +
        ph_score * weights["ph"]
    ) * 100

    return round(final_score, 2)


# ---------------------------------------------------------
# MAIN API ROUTE
# ---------------------------------------------------------
@router.get("/recommend_crop")
async def recommend_crop(
    N: float, P: float, K: float,
    temperature: float, humidity: float,
    ph: float, rainfall: float,
    city: str
):

    # 1️⃣ Model input (simple — rainfall comes directly from the user)
    X = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Soil used for scoring
    soil = {
        "N": float(N),
        "P": float(P),
        "K": float(K),
        "temperature": float(temperature),
        "humidity": float(humidity),
        "ph": float(ph),
        "rainfall": float(rainfall)
    }

    # 2️⃣ Fetch weather (RapidAPI)
    weather = fetch_weather(city)

    # 3️⃣ ML Probabilities
    probs = model.predict_proba(X)[0]
    top_idx = np.argsort(probs)[::-1]

    scored_crops = []

    for i in top_idx[:4]:
        class_label = model.classes_[i]
        crop_name = encoder.inverse_transform([class_label])[0]
        req = crop_reqs.get(crop_name.lower(), {})

        suitability = score_crop(req, weather, soil)
        final_score = 0.7 * suitability + 0.3 * (float(probs[i]) * 100)

        scored_crops.append({
            "crop": crop_name,
            "suitability": float(suitability),
            "model_prob": float(probs[i]),
            "final_score": float(final_score)
        })

    scored_crops = sorted(scored_crops, key=lambda x: x["final_score"], reverse=True)
    best_crop = scored_crops[0]["crop"]

    return {
        "recommended_crop": best_crop,
        "weather": weather,
        "ranked_crops": scored_crops
    }
