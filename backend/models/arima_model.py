# backend/models/arima_model.py
import pandas as pd
import statsmodels.api as sm

# Use the uploaded dataset path
DATA_PATH = "data/agriculture_time_series_dataset.csv"
# If you prefer to keep dataset inside repo, copy the CSV to backend/data/ and change path.

def load_time_series():
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def predict_rainfall_next_month():
    df = load_time_series()
    series = df['rainfall_mm'].astype(float).dropna()

    if len(series) < 6:
        # fallback: not enough data; return last observed
        return float(series.iloc[-1])

    model = sm.tsa.ARIMA(series, order=(1,1,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=1)
    return float(forecast.iloc[0])

# Optional multi-step forecast
def forecast_rainfall_steps(steps=12):
    df = load_time_series()
    series = df['rainfall_mm'].astype(float).dropna()
    model = sm.tsa.ARIMA(series, order=(1,1,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast.tolist()
