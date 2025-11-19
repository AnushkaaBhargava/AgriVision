import streamlit as st
import requests
from PIL import Image
import io
import requests

from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("API_KEY")

CITY = "Greater Noida"
UNITS = "metric"                         # Celsius

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units={UNITS}"
    
    try:
        r = requests.get(url)
        data = r.json()

        if data.get("cod") != 200:
            return None
        
        return {
            "temp": data["main"]["temp"],
            "temp_min": data["main"]["temp_min"],
            "temp_max": data["main"]["temp_max"],
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"].title()
        }

    except:
        return None


# ----------------------------------------------------------
# PAGE CONFIG + FORCE LIGHT MODE
# ----------------------------------------------------------
st.set_page_config(page_title="AgriVision", page_icon="🌱", layout="wide")

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

            /* FIX INPUT LABELS NOT VISIBLE */
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
    border: none !important;
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

.alert-card {
    background: #FFF8E7;
    padding: 18px;
    border-radius: 12px;
    border-left: 6px solid #F2A83A;
}

.score-box {
    background: white;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 7px 18px rgba(0,0,0,0.07);
}

.score-num {
    color: #3F6B2F;
    font-size: 48px;
    font-weight: 900;
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
    color: white !important;
}
            
            
            

</style>
""", unsafe_allow_html=True)




# ----------------------------------------------------------
# TOP NAVIGATION BAR
# ----------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

menu_items = ["Dashboard", "Crop Recommendation", "Disease Detection", "Fertilizer Advice"]

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
        weather = get_weather(CITY)
        if weather:
            st.header(f"🌤 Weather Today: {weather['temp']}°C")
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
    st.write("Enter the soil and climate details below:")
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        N = st.number_input("Nitrogen (N)", 0, 200, 50)
        P = st.number_input("Phosphorus (P)", 0, 200, 50)

    with col2:
        K = st.number_input("Potassium (K)", 0, 200, 50)
        temperature = st.number_input("Temperature (°C)", 0, 50, 25)

    with col3:
        humidity = st.number_input("Humidity (%)", 0, 100, 70)
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
        rainfall = st.number_input("Rainfall (mm)", 0, 500, 100)

    if st.button("🌱 Recommend Crop"):

        params = {
            "N": N, "P": P, "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
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
# FERTILIZER ADVICE PAGE (Simple Query Param Version)
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
