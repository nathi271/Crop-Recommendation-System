import streamlit as st
import joblib
import numpy as np

# Load the three saved files (must be in the same folder)
model = joblib.load('crop_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

st.set_page_config(page_title="Crop Recommender TN", layout="wide")

st.title("🌾 Tamil Nadu Crop Recommendation System")
st.markdown("**Accuracy: ~99.55%** (Random Forest model)")
st.write("Enter soil nutrients and weather details to get the best crop suggestion.")

col1, col2 = st.columns(2)

with col1:
    N = st.slider("Nitrogen (N)", 0, 140, 50)
    P = st.slider("Phosphorus (P)", 5, 145, 50)
    K = st.slider("Potassium (K)", 5, 205, 50)
    temp = st.slider("Temperature (°C)", 8.0, 44.0, 25.0)

with col2:
    humidity = st.slider("Humidity (%)", 14.0, 100.0, 70.0)
    ph = st.slider("pH Value", 3.5, 10.0, 7.0)
    rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0)

if st.button("Get Recommended Crop", type="primary"):
    input_data = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)
    pred_encoded = model.predict(input_scaled)
    crop = le.inverse_transform(pred_encoded)[0]

    st.success(f"**Recommended Crop: {crop.capitalize()}** 🌱")
    st.balloons()