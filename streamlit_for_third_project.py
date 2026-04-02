import streamlit as st
import base64
 
st.set_page_config(layout="wide")
 
def get_base64_image(image_path):
     with open(image_path, "rb") as f:
         data = f.read()
     return base64.b64encode(data).decode()
 
image_path = r"C:\Users\user\Documents\guvi\guvi project 1\third project\bg_taxi.png"
encoded = get_base64_image(image_path)
 
 # Inject CSS to add the background
st.markdown(
     f"""
     <style>
     .stApp {{
         background-image: url("data:image/webp;base64,{encoded}");
         background-size: cover;
         background-repeat: no-repeat;
         background-attachment: fixed;
         background-position: center;
     }}
     </style>
     """,
     unsafe_allow_html=True
)
st.markdown("""
    <div style="
        background-color: black;
        border-radius: 16px;
        padding: 12px 6px;
        margin-bottom: 20px;
        text-align: center;
        display: inline-block;
        width: 100%;
    ">
        <h1 style="
            color: white;
            -webkit-text-stroke: 2px blue;
            font-weight: bold;
            margin: 0;
        ">
            🚕 Urban City Taxi Fare Prediction 🚕
        </h1>
    </div>
""", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

# Load saved pipeline (preprocessing + model)
pipeline = joblib.load(r"C:\Users\user\Documents\guvi\guvi project 1\third project\final_pipeline.joblib")

st.header("Enter trip details")

# User inputs
vendor_id = st.selectbox("Vendor ID", options=[1, 2])
passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)

pickup_date = st.date_input("Pickup Date", value=datetime.today().date())
pickup_time = st.time_input("Pickup Time", value=datetime.now().time())
pickup_longitude = st.number_input("Pickup Longitude", format="%.6f", value=-73.985428)
pickup_latitude = st.number_input("Pickup Latitude", format="%.6f", value=40.748817)
dropoff_longitude = st.number_input("Dropoff Longitude", format="%.6f", value=-73.985428)
dropoff_latitude = st.number_input("Dropoff Latitude", format="%.6f", value=40.748817)
ratecode_id = st.selectbox("Rate Code ID", options=[1, 2, 3, 4, 5, 6])
extra = st.number_input("Extra Charges", value=0.5)
mta_tax = st.number_input("MTA Tax", value=0.5)
tip_amount = st.number_input("Tip Amount", value=0.0)
tolls_amount = st.number_input("Tolls Amount", value=0.0)
improvement_surcharge = st.number_input("Improvement Surcharge", value=0.3)
fare_amount = st.number_input("Fare Amount", value=5.0)

# Combine date and time into datetime
pickup_datetime = datetime.combine(pickup_date, pickup_time)

# Compute datetime-based features
hour = pickup_datetime.hour
pickup_day = pickup_datetime.weekday()  # Monday=0
is_weekend = 1 if pickup_day >= 5 else 0
is_night = 1 if (hour >= 22 or hour <= 5) else 0

# Define haversine function for distance calculation
def haversine(lon1, lat1, lon2, lat2):
    from math import radians, cos, sin, asin, sqrt
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth radius in km
    return c * r

trip_distance = haversine(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude)

# Compute log features safely
trip_distance_log = np.log1p(trip_distance)
fare_amount_log = np.log1p(fare_amount)

# Build input DataFrame with all expected features by your pipeline
input_dict = {
    'passenger_count': passenger_count,
    'pickup_longitude': pickup_longitude,
    'pickup_latitude': pickup_latitude,
    'RatecodeID': ratecode_id,
    'dropoff_longitude': dropoff_longitude,
    'dropoff_latitude': dropoff_latitude,
    'extra': extra,
    'mta_tax': mta_tax,
    'tip_amount': tip_amount,
    'tolls_amount': tolls_amount,
    'improvement_surcharge': improvement_surcharge,
    'hour': hour,
    'is_weekend': is_weekend,
    'is_night': is_night,
    'trip_distance': trip_distance,
    'trip_distance_log': trip_distance_log,
    'fare_amount_log': fare_amount_log
}

input_df = pd.DataFrame([input_dict])

if st.button("Predict Fare"):
    prediction = pipeline.predict(input_df)[0]
    st.success(f"Estimated Total Fare: ${prediction:.2f}")
