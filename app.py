import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# ---------------------- LOAD MODEL -----------------------
model = load_model('heart_disease_model.h5')
scaler = joblib.load('scaler.pkl')

# ---------------------- CUSTOM CSS ------------------------
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #d32f2f;
            text-align: center;
            padding: 10px;
        }
        .subtitle {
            font-size: 18px;
            color: #616161;
            text-align: center;
            margin-bottom: 20px;
        }
        .input-card {
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        }
        .prediction-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            margin-top: 20px;
            box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- HEADER -------------------------
st.markdown("<div class='title'>❤️ Heart Disease Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered health prediction using ANN</div>", unsafe_allow_html=True)

# ---------------------- SIDEBAR -------------------------
st.sidebar.title("📌 About This App")
st.sidebar.info("""
This app predicts **Heart Disease Risk** using:
- ANN Model  
- Feature Scaling  
- User Inputs  

Created with **Streamlit + TensorFlow**.
""")
st.sidebar.markdown("___")
st.sidebar.write("""
### 👨‍💻 Developed by:
**Dharmesh Kushwaha (Team Leader)**

### 👥 Team Members:
- Nitin Verma  
- Mandeep Kumar  
- Ronit Maurya  
- Shivam Maurya
- Khyati Singh
- Nishtha Agarwal
- Abhay Mishra 
""")

# ---------------------- INPUT CARD -------------------------
st.markdown("<div class='input-card'>", unsafe_allow_html=True)

st.subheader("🧍 Patient Health Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🧓 Age", min_value=1, max_value=120, value=30)
    blood_pressure = st.number_input("🩸 Blood Pressure", min_value=50, max_value=200, value=120)
    bmi = st.number_input("⚖️ Body Mass Index (BMI)", min_value=10.0, max_value=70.0, value=25.0)

with col2:
    gender = st.selectbox("⚧ Gender", ["Male", "Female"])
    cholesterol = st.number_input("🧈 Cholesterol Level", min_value=100, max_value=400, value=200)
    glucose_level = st.number_input("🍬 Glucose Level", min_value=50, max_value=300, value=100)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- PROCESS INPUT -------------------------
gender_binary = 1 if gender == "Male" else 0

input_data = pd.DataFrame({
    "age": [age],
    "blood_pressure": [blood_pressure],
    "cholesterol": [cholesterol],
    "bmi": [bmi],
    "glucose_level": [glucose_level],
    "gender": [gender_binary]
})

scaled_input = scaler.transform(input_data)

# ---------------------- PREDICTION -------------------------
if st.button("🔍 Predict Heart Disease Risk"):
    prediction = model.predict(scaled_input)
    result = (prediction > 0.5).astype(int)[0][0]

    st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)

    if result == 1:
        st.error("🚨 **High Chance of Heart Disease**")
    else:
        st.success("💚 **Low Chance of Heart Disease**")

    st.markdown("</div>", unsafe_allow_html=True)
