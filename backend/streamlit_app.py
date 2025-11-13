import streamlit as st
import requests
from PIL import Image
import io

# -------------------
# CONFIG
# -------------------
BACKEND_URL = "http://127.0.0.1:8000"  # Change if deployed

st.set_page_config(page_title="AgriVision", page_icon="🌾", layout="centered")

# -------------------
# SIDEBAR
# -------------------
st.sidebar.title("🌱 AgriVision")
st.sidebar.write("AI-based Crop Recommendation and Disease Detection System")
st.sidebar.divider()
page = st.sidebar.radio("Select Feature", ["Crop Recommendation 🌾", "Disease Detection 🍃"])

# -------------------
# CROP RECOMMENDATION
# -------------------
if page == "Crop Recommendation 🌾":
    st.header("🌾 Crop Recommendation System")
    st.write("Enter soil and environmental details:")

    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=50)
        P = st.number_input("Phosphorus (P)", min_value=0, max_value=140, value=50)
        K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
    with col2:
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
    with col3:
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

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
            crop_name = response.json().get("recommended_crop", "Unknown")
            st.success(f"✅ Recommended Crop: **{crop_name}**")
        else:
            st.error("❌ Something went wrong. Please try again.")

# -------------------
# DISEASE DETECTION
# -------------------
elif page == "Disease Detection 🍃":
    st.header("🍃 Plant Disease Detection")
    st.write("Upload a leaf image to detect disease:")

    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf", use_column_width=True)

        if st.button("🔍 Detect Disease"):
            with st.spinner("Analyzing image..."):
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(f"{BACKEND_URL}/disease/detect_disease", files=files)

            if response.status_code == 200:
                result = response.json()
                disease = result.get("disease", "Unknown")
                confidence = result.get("confidence", 0.0)
                st.success(f"🩺 Disease: **{disease}**")
                st.info(f"Confidence: **{confidence:.2f}%**")
            else:
                st.error("❌ Failed to detect disease. Try again.")
