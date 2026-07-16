import streamlit as st
import requests
from PIL import Image
import io
import requests
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()
API_KEY = os.getenv("API_KEY")

CITY = "Greater Noida"
UNITS = "metric"  # Celsius


# ----------------------------------------------------------
# WEATHER API FUNCTION
# ----------------------------------------------------------
def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"

    try:
        r = requests.get(url)
        data = r.json()


        if data.get("cod") != "200":
            return None

        today = data.get("list", [])[:8]
        if not today:
            return None

        def safe(item, key):
            return item.get(key) if isinstance(item, dict) else None

        temps = []
        descriptions = []

        for i in today:
            main = i.get("main", {})
            temps.append(safe(main, "temp"))

            weather_desc = i.get("weather", [{}])[0].get("description", "")
            descriptions.append(weather_desc)

        temps_clean = [t for t in temps if t is not None]

        return {
            "temp": temps_clean[0] if temps_clean else None,
            "temp_min": min(temps_clean) if temps_clean else None,
            "temp_max": max(temps_clean) if temps_clean else None,
            "humidity": today[0].get("main", {}).get("humidity"),
            "description": max(set(descriptions), key=descriptions.count) if descriptions else ""
        }

    except Exception as e:
        st.write("Error fetching weather:", e)
        return None



# ----------------------------------------------------------
# PAGE CONFIG + FORCE LIGHT MODE
# ----------------------------------------------------------
st.set_page_config(page_title="AgriVision", page_icon="🌱", layout="wide")

if "selected_city" not in st.session_state:
    st.session_state.selected_city = "Greater Noida"

st.markdown("""
<style>
/* FORCE LIGHT MODE */
[data-testid="stAppViewContainer"] {
    background-color: #F4F6EE !important;
    color: black !important;
}
[data-testid="stHeader"] {
    background-color: #F4F6EE !important;
}
[data-testid="stToolbar"] {
    background-color: #F4F6EE !important;
}

/* Label fixes */
label, .stNumberInput label, .stTextInput label, .stSelectbox label {
    color: #3B3B3B !important;
    font-weight: 600 !important;
}

/* NAVBAR */
.top-nav {
    width: 100%;
    background-color: #E9F2D9;
    padding: 14px 0;
    border-bottom: 2px solid #DADFCC;
    display: flex;
    justify-content: center;
    gap: 45px;
    position: sticky;
    top: 0;
    z-index: 999;
}
.nav-btn {
    padding: 10px 22px;
    background: #F5F8EC;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 600;
    color: #3F5F32;
    border: 1px solid #D6E3C5;
    cursor: pointer;
}
.nav-btn:hover {
    background-color: #DCE8C7;
}
.active-nav {
    background-color: #4E6B37 !important;
    color: white !important;
}

/* Cards */
.agri-card {
    background: #FFFFFF;
    padding: 10px 15px;
    border-radius: 18px;
    box-shadow: 0px 8px 22px rgba(0,0,0,0.06);
    margin-bottom: 25px;
}

/* Fert Cards */
.fert-card {
    background: #F5F8EC;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #DCE8C8;
    margin-bottom: 12px;
}

/* Disease Result */
.result-card {
    background: #FFF4D7;
    padding: 22px;
    border-radius: 16px;
    border-left: 8px solid #E5A437;
    margin-top: 15px;
    margin-bottom: 20px;
}

div.stButton > button {
    background-color: #4E6B37 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    box-shadow: 0px 3px 8px rgba(0,0,0,0.15);
}
div.stButton > button:hover {
    background-color: #365026 !important;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------
# TOP NAVIGATION BAR
# ----------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

menu_items = ["Dashboard", "Crop Recommendation", "Disease Detection", "Fertilizer Advice","Rainfall"]

st.markdown("<div class='top-nav'>", unsafe_allow_html=True)
cols = st.columns(len(menu_items))

for i, item in enumerate(menu_items):
    if item == st.session_state.page:
        cols[i].button(item, key=item, help=item)
        st.markdown("<style>div[data-testid='stButton'] button {background:#4E6B37;color:white;}</style>",
                    unsafe_allow_html=True)
    else:
        if cols[i].button(item, key=item):
            st.session_state.page = item
st.markdown("</div>", unsafe_allow_html=True)


BACKEND_URL = "http://127.0.0.1:8000"
page = st.session_state.page


# =================================================================
# DASHBOARD PAGE
# =================================================================
if page == "Dashboard":
    st.title("🌾 Welcome to AgriVision Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='agri-card'>", unsafe_allow_html=True)

        weather = get_weather(st.session_state.selected_city)

        if weather:
            st.header(f"🌤 Weather in {st.session_state.selected_city}: {weather['temp']}°C")
            st.write(f"Min: {weather['temp_min']}°C • Max: {weather['temp_max']}°C")
            st.write(f"Condition: {weather['description']}")
        else:
            st.header("🌤 Weather Today: N/A")
            st.write("⚠ Unable to load live weather data.")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='agri-card'>", unsafe_allow_html=True)
        st.header("👋 Hello Farmer!")
        st.write("Here are today's insights for your crops.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='agri-card'>", unsafe_allow_html=True)
    st.subheader("🌿 Best Crop Today: Cotton")
    st.write("Yield: 85% • Profit: 92% • Sustainability: 78%")
    st.markdown("</div>", unsafe_allow_html=True)


# =================================================================
# CROP RECOMMENDATION PAGE
# =================================================================
if page == "Crop Recommendation":

    st.markdown("<div class='agri-card'>", unsafe_allow_html=True)
    st.header("🌾 Crop Recommendation")
    st.write("Enter soil details. Weather (temperature + humidity) is auto–fetched from the selected city.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------
    # City Selection
    # ----------------------
    CITY_LIST = [
        "Greater Noida", "Mumbai", "Delhi", "Bangalore", "Chennai",
        "Kolkata", "Hyderabad", "London", "Tokyo", "New York"
    ]

    city = st.selectbox(
        "Select City for Live Weather",
        CITY_LIST,
        index=CITY_LIST.index(st.session_state.selected_city)
        if st.session_state.selected_city in CITY_LIST else 0
    )

    st.session_state.selected_city = city

    # Fetch live weather
    weather = get_weather(city)


    # Auto-fill values safely
    auto_temp = float(weather.get("temp")) if (weather and weather.get("temp")) else 25.0
    auto_humidity = float(weather.get("humidity")) if (weather and weather.get("humidity")) else 60.0

    # ----------------------
    # INPUT BOXES (RESTORED)
    # ----------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
        P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)

    with col2:
        K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=auto_temp)

    with col3:
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=auto_humidity)
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
        rainfall = st.number_input("Rainfall (mm)", min_value=0, max_value=500, value=100)

    # ----------------------
    # Submit button
    # ----------------------
    if st.button("🌱 Recommend Crop"):

        params = {
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
            "city": city
        }

        with st.spinner("Predicting best crop..."):
            response = requests.get(f"{BACKEND_URL}/crop/recommend_crop", params=params)

        if response.status_code == 200:
            result = response.json()
            crop = result.get("recommended_crop", "Unknown")

            st.markdown("<div class='agri-card'>", unsafe_allow_html=True)
            st.subheader(f"🌿 Recommended Crop: **{crop}**")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("❌ Failed to fetch recommendation.")
            st.write(response.text)

# =================================================================
# DISEASE DETECTION PAGE
# =================================================================
if page == "Disease Detection":

    st.title("🍁 Plant Disease Detection")

    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf", width=300)

        if st.button("🔍 Detect Disease"):

            files = {"file": uploaded_file.getvalue()}

            with st.spinner("Analyzing image..."):
                response = requests.post(f"{BACKEND_URL}/disease/detect_disease", files=files)

            if response.status_code == 200:
                result = response.json()
                disease = result.get("disease")
                confidence = result.get("confidence")

                st.markdown(f"""
                <div class='result-card'>
                    <h3>🌿 Detected Disease: {disease}</h3>
                    <p><b>Confidence: {confidence:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.error("❌ Failed to detect disease.")


# =================================================================
# FERTILIZER ADVICE PAGE
# =================================================================
if page == "Fertilizer Advice":

    st.title("🧪 Fertilizer Recommendations")

    col1, col2, col3 = st.columns(3)

    with col1:
        N = st.number_input("Nitrogen (N)", 0.0, 300.0, 90.0)
        P = st.number_input("Phosphorus (P)", 0.0, 300.0, 40.0)

    with col2:
        K = st.number_input("Potassium (K)", 0.0, 300.0, 45.0)
        temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0)

    with col3:
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 1000.0, 200.0)

    if st.button("🔍 Get Fertilizer Advice"):

        params = {
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        }

        with st.spinner("Fetching fertilizer advice..."):
            response = requests.post(f"{BACKEND_URL}/fertilizer/predict", params=params)

        if response.status_code == 200:
            result = response.json()

            st.success("Fertilizer recommendation ready!")

            st.markdown(f"""
                <div class='fert-card'>
                    <b>🌱 Fertilizer Package:</b><br>{result['fertilizer_package']}
                </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <div class='fert-card'>
                    <b>📦 Dosage:</b> {result['dosage_kg_per_acre_or_tree']}
                </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <div class='fert-card'>
                    <b>⏳ Application Frequency:</b> {result['application_frequency']}
                </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <div class='fert-card'>
                    <b>📝 Notes:</b><br>{result['recommendation_notes']}
                </div>
            """, unsafe_allow_html=True)

        else:
            st.error("❌ Failed to fetch fertilizer recommendation.")
            st.write(response.text)


if page == "Rainfall":

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from sklearn.metrics import mean_squared_error
    from pathlib import Path

    st.header("🌧 Rainfall Forecast – Complete Time Series Analysis")

    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "data" / "agriculture_time_series_dataset.csv"

    # Load data
    df = pd.read_csv(DATA_PATH)
    try:
        # Load data
        df = pd.read_csv(DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        rainfall = df['rainfall_mm']

        st.subheader("📉 Historical Rainfall")
        st.line_chart(rainfall)

        # ============= Seasonal Decomposition =============
        st.subheader("📊 Seasonal Decomposition (Trend | Seasonality | Residual)")
        try:
            decomposition = seasonal_decompose(rainfall, model='additive', period=12)

            fig, ax = plt.subplots(4, 1, figsize=(10, 8))
            decomposition.observed.plot(ax=ax[0], title="Observed")
            decomposition.trend.plot(ax=ax[1], title="Trend")
            decomposition.seasonal.plot(ax=ax[2], title="Seasonality")
            decomposition.resid.plot(ax=ax[3], title="Residuals")
            st.pyplot(fig)
        except Exception:
            st.warning("⚠ Could not perform seasonal decomposition due to insufficient data.")


        # ============= Forecast Input =============
        months = st.number_input(
            "📅 Forecast next N months:",
            min_value=1,
            max_value=36,
            value=12
        )

        if st.button("🔮 Run ARIMA Forecast"):

            with st.spinner("Training ARIMA(1,1,1) model…"):

                model = sm.tsa.ARIMA(rainfall, order=(1, 1, 1))
                model_fit = model.fit()

                forecast = model_fit.forecast(steps=months)

            # Future dates
            future_dates = pd.date_range(
                start=rainfall.index[-1],
                periods=months + 1,
                freq="M"
            )[1:]

            # Combine forecast
            forecast_series = pd.Series(forecast, index=future_dates)

            # ============= Plot Forecast =============
            st.subheader("📈 Forecasted Rainfall")
            fig2, ax = plt.subplots()
            ax.plot(rainfall, label="Historical")
            ax.plot(forecast_series, label="Forecast")
            ax.legend()
            st.pyplot(fig2)

            # ============= Metrics (RMSE, MAPE) =============
            st.subheader("📏 Model Performance")

            # Last N months true values (simple comparison)
            try:
                test_true = rainfall[-months:]
                test_pred = forecast[:len(test_true)]

                rmse = np.sqrt(mean_squared_error(test_true, test_pred))

                mape = np.mean(np.abs((test_true - test_pred) / test_true)) * 100

                st.write(f"**RMSE:** {rmse:.2f}")
                st.write(f"**MAPE:** {mape:.2f}%")
            except:
                st.write("⚠ Not enough data for performance metrics.")

            # ============= Download Forecast =============
            st.subheader("⬇ Download Forecast Results")

            csv_data = pd.DataFrame({
                "date": future_dates,
                "forecast_rainfall_mm": forecast
            }).to_csv(index=False)

            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv_data,
                file_name="rainfall_forecast.csv",
                mime="text/csv"
            )

            # ============= Insights =============
            st.subheader("🧠 Automatic Insights")

            avg_future = np.mean(forecast)
            avg_past = np.mean(rainfall[-12:])

            if avg_future > avg_past:
                st.success("🌧 Higher rainfall predicted — good for Rice, Jute, Sugarcane.")
            else:
                st.info("🌤 Lower rainfall predicted — suitable for Wheat, Chickpea, Maize.")

            st.write("Raw Forecast Values:")
            st.write(forecast_series)

    except Exception as e:
        st.error("⚠ Could not load dataset or run analysis.")
        st.write(e)
