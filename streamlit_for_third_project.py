# -------------------------------
# IMPORTS
# -------------------------------
import streamlit as st
import base64
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Taxi Fare Prediction", layout="wide")

# -------------------------------
# SESSION STATE
# -------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# -------------------------------
# BACKGROUND IMAGE
# -------------------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

image_path = r"C:\Users\user\Documents\guvi\guvi project 1\third project\bg_taxi.png"
encoded = get_base64_image(image_path)

st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/png;base64,{encoded}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load(r"C:\Users\user\Documents\guvi\guvi project 1\third project\tripfare_model.pkl")

pipeline = load_model()

# =========================================================
# 🏠 HOME PAGE
# =========================================================
if st.session_state.page == "home":

    st.markdown("""
    <h1 style="
            color: white;
            -webkit-text-stroke: 2px blue;
            text-align:center;
            font-weight: bold;
            margin: 0;
        ">
            🚕  Urban City Taxi Fare Prediction 🚕
        </h1>
    """, unsafe_allow_html=True)

    st.markdown("### 👉 Slide the taxi to start your trip")

    position = st.slider("", 0, 100, 0)

    st.markdown(f"""
    <div style="position: relative; height: 120px;">
        <div style="position: absolute; left: {position}%; font-size: 60px;">
            🚖
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:5px;background:white;margin-top:-30px;'></div>", unsafe_allow_html=True)

    if position > 99:
        if st.button("🚀 Enter Trip Details"):
            st.session_state.page = "predict"
            st.rerun()

# =========================================================
# 📊 PREDICTION PAGE
# =========================================================
elif st.session_state.page == "predict":

    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("""
    <h2 style='color:white;text-shadow:0 0 8px blue;'>
    Enter Trip Details
    </h2>
    """, unsafe_allow_html=True)

    # -------------------------------
    # INPUTS
    # -------------------------------
    col1, col2 = st.columns(2)

    with col1:
        passenger_count = st.number_input("Passengers", 1, 6, 1)
        ratecode_id = st.selectbox("Rate Code", [1, 2, 3, 4, 5, 6])

    with col2:
        pickup_date = st.date_input("Pickup Date")
        pickup_time = st.time_input("Pickup Time")

    # -------------------------------
    # LOCATION
    # -------------------------------
    st.markdown("### 📍 Location Details")

    col3, col4 = st.columns(2)

    with col3:
        pickup_longitude = st.number_input("Pickup Longitude", format="%.6f", value=-73.985428)
        pickup_latitude = st.number_input("Pickup Latitude", format="%.6f", value=40.748817)

    with col4:
        dropoff_longitude = st.number_input("Dropoff Longitude", value=-73.981)
        dropoff_latitude = st.number_input("Dropoff Latitude", value=40.752)

    # -------------------------------
    # FEATURE ENGINEERING
    # -------------------------------
    pickup_datetime = datetime.combine(pickup_date, pickup_time)

    hour = pickup_datetime.hour
    pickup_day = pickup_datetime.weekday()

    is_weekend = 1 if pickup_day >= 5 else 0
    is_night = 1 if (hour >= 22 or hour <= 5) else 0

    # Haversine Distance
    def haversine(lon1, lat1, lon2, lat2):
        from math import radians, cos, sin, asin, sqrt
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return 6371 * c

    trip_distance = haversine(
        pickup_longitude, pickup_latitude,
        dropoff_longitude, dropoff_latitude
    )

    # -------------------------------
    # VALIDATION
    # -------------------------------
    if trip_distance <= 0:
        st.error("Invalid trip distance. Please check coordinates.")

    # -------------------------------
    # BUILD INPUT (MATCH TRAINING)
    # -------------------------------
    input_dict = {
        'passenger_count': passenger_count,
        'pickup_longitude': pickup_longitude,
        'pickup_latitude': pickup_latitude,
        'dropoff_longitude': dropoff_longitude,
        'dropoff_latitude': dropoff_latitude,
        'RatecodeID': ratecode_id,
        'hour': hour,
        'is_weekend': is_weekend,
        'is_night': is_night,
        'trip_distance': trip_distance
    }

    input_df = pd.DataFrame([input_dict])

    input_df = input_df[pipeline.feature_names_in_]

    # -------------------------------
    # PREDICTION
    # -------------------------------
    if st.button("🚀 Predict Fare"):
        try:
            input_df = pd.DataFrame([input_dict])

             # 🔥 Ensure exact feature match & order
            input_df = input_df[pipeline.feature_names_in_]

            prediction = pipeline.predict(input_df)[0]

            st.markdown(f"""
            ### 💰 Estimated Fare:
            # **${prediction:.2f}**
        """)

            st.info("Prediction based on distance, time, and trip conditions.")

        except Exception as e:
            st.error(f"Error: {e}")

        except Exception as e:
            st.error(f"Error: {e}")