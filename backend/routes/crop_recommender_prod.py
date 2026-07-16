#!/usr/bin/env python3
import argparse
import json
import logging
import os
import pickle
import sys
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from models.arima_model import predict_rainfall_next_month


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# -----------------------
# Predefined Cities
# -----------------------
CITY_LIST = {
    "greater_noida": "Greater Noida",
    "new_york": "New York",
    "mumbai": "Mumbai",
    "delhi": "Delhi",
    "bangalore": "Bangalore",
    "chennai": "Chennai",
    "kolkata": "Kolkata",
    "hyderabad": "Hyderabad",
    "tokyo": "Tokyo",
    "london": "London",
    "toronto": "Toronto",
    "sydney": "Sydney",
    "dubai": "Dubai",
    "singapore": "Singapore",
    "paris": "Paris",
    "berlin": "Berlin",
    "los_angeles": "Los Angeles",
    "san_francisco": "San Francisco",
    "seattle": "Seattle"
}


# -----------------------
# Model Loaders
# -----------------------
def load_xgb_model(path: str) -> xgb.XGBClassifier:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    if path.endswith('.pkl'):
        with open(path, "rb") as f:
            return pickle.load(f)

    model = xgb.XGBClassifier()
    model.load_model(path)
    return model


def load_label_encoder(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label encoder not found: {path}")
    return pickle.load(open(path, "rb"))


def parse_feature_values(csv_str: str) -> np.ndarray:
    vals = [float(x.strip()) for x in csv_str.split(",")]
    if len(vals) != 7:
        raise ValueError("Expected 7 values: N,P,K,temp,humidity,pH,rainfall")
    return np.array(vals).reshape(1, -1)


def load_feature_row_from_csv(path: str, index: int = 0) -> np.ndarray:
    df = pd.read_csv(path)
    row = df.iloc[index].values.astype(float)
    return row.reshape(1, -1)


# -----------------------
# Weather via RapidAPI
# -----------------------
def fetch_weather_rapidapi(api_key: str, city_name: str):
    url = "https://open-weather13.p.rapidapi.com/city"
    headers = {
        "x-rapidapi-host": "open-weather13.p.rapidapi.com",
        "x-rapidapi-key": api_key
    }
    params = {"city": city_name, "lang": "EN"}

    resp = requests.get(url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    temp_raw = data["main"]["temp"]
    temp_c = (temp_raw - 32) * 5 / 9 if temp_raw > 60 else temp_raw

    humidity = data["main"]["humidity"]
    wind = data.get("wind", {}).get("speed", 2.0)

    monthly_rain = 150  # RapidAPI does not give rainfall

    return {
        "avg_temp_c": round(temp_c, 2),
        "avg_humidity_pct": humidity,
        "avg_wind_m_s": wind,
        "avg_daily_precip_mm": monthly_rain / 30,
        "monthly_precip_mm_est": monthly_rain,
        "annual_precip_mm_est": monthly_rain * 12,
        "raw": data
    }


# -----------------------
# Crop Scoring Logic
# -----------------------
def score_crop(requirements, weather, soil):
    reasons = []
    weights = requirements.get(
        "weights",
        {"temp": 0.35, "rain": 0.35, "humidity": 0.15, "ph": 0.15}
    )

    # Temperature
    temp_score = 0.6
    if "temp_min" in requirements:
        t = weather["avg_temp_c"]
        if requirements["temp_min"] <= t <= requirements["temp_max"]:
            temp_score = 1.0
        else:
            dist = min(abs(t - requirements["temp_min"]), abs(t - requirements["temp_max"]))
            temp_score = max(0, 1 - dist / 10)

    # Rainfall
    rain_score = 0.6
    monthly = weather["monthly_precip_mm_est"]
    if "monthly_rain_min" in requirements:
        if requirements["monthly_rain_min"] <= monthly <= requirements["monthly_rain_max"]:
            rain_score = 1.0
        else:
            dist = min(abs(monthly - requirements["monthly_rain_min"]),
                       abs(monthly - requirements["monthly_rain_max"]))
            rain_score = max(0, 1 - dist/100)

    # Humidity
    hum_score = 0.6
    if "humidity_min" in requirements:
        h = weather["avg_humidity_pct"]
        if requirements["humidity_min"] <= h <= requirements["humidity_max"]:
            hum_score = 1.0
        else:
            dist = min(abs(h - requirements["humidity_min"]),
                       abs(h - requirements["humidity_max"]))
            hum_score = max(0, 1 - dist / 50)

    # pH
    ph_score = 0.6
    if "ph_min" in requirements:
        soil_ph = soil["ph"]
        if requirements["ph_min"] <= soil_ph <= requirements["ph_max"]:
            ph_score = 1.0
        else:
            dist = min(abs(soil_ph - requirements["ph_min"]),
                       abs(soil_ph - requirements["ph_max"]))
            ph_score = max(0, 1 - dist / 3)

    final_score = (
        temp_score * weights["temp"] +
        rain_score * weights["rain"] +
        hum_score * weights["humidity"] +
        ph_score * weights["ph"]
    ) * 100

    return round(final_score, 2)


# -----------------------
# Prediction
# -----------------------
def top_k_predictions(model, X, encoder, k=4):
    probs = model.predict_proba(X)[0]
    idx = np.argsort(probs)[::-1][:k]
    out = []

    for i in idx:
        class_label = model.classes_[i]
        crop_name = encoder.inverse_transform([class_label])[0]
        out.append((crop_name, float(probs[i])))

    return out


# -----------------------
# MAIN
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="RapidAPI key")
    parser.add_argument("--city", required=True, choices=list(CITY_LIST.keys()))
    parser.add_argument("--feature-values")
    parser.add_argument("--features-file")
    parser.add_argument("--feature-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=4)

    args = parser.parse_args()

    model_path = "models/XGBoost-final-crop.pkl"
    encoder_path = "models/label_encoder.pkl"
    crop_req_path = "data/crop_requirements.json"

    model = load_xgb_model(model_path)
    encoder = load_label_encoder(encoder_path)
    crop_reqs = json.load(open(crop_req_path, "r"))

    if args.features_file:
        X = load_feature_row_from_csv(args.features_file, args.feature_index)
    else:
        X = parse_feature_values(args.feature_values)

    soil_keys = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    soil = {k: float(v) for k, v in zip(soil_keys, X.flatten().tolist())}

    city_name = CITY_LIST[args.city]
    weather = fetch_weather_rapidapi(args.api_key, city_name)

    predictions = top_k_predictions(model, X, encoder, args.top_k)

    final_list = []
    for crop, prob in predictions:
        req = crop_reqs.get(crop.lower(), {})
        suitability = score_crop(req, weather, soil)
        final_score = round(0.7 * suitability + 0.3 * (prob * 100), 2)

        final_list.append({
            "crop": crop,
            "model_prob": prob,
            "suitability_score": suitability,
            "final_rank_score": final_score
        })

    final_list = sorted(final_list, key=lambda x: x["final_rank_score"], reverse=True)

    print(json.dumps({
        "city": city_name,
        "weather_summary": {k: v for k, v in weather.items() if k != "raw"},
        "soil": soil,
        "results": final_list,
        "recommended_top2": final_list[:2]
    }, indent=2))


if __name__ == "__main__":
    main()
