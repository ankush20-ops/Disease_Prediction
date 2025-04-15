import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load trained model
model = joblib.load(r"C:\Users\jaiba\Downloads\capstone\random_forest_heart_model.pkl")

st.set_page_config(page_title="AI Heart Health Assistant", layout="centered")
st.title("❤️ AI Heart Disease Risk Predictor")
st.markdown("""
This app predicts the likelihood of **heart disease** based on medical inputs.
""")

st.sidebar.header("User Health Inputs")

def user_input_features():
    age = st.sidebar.slider('Age', 20, 90, 45)
    sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
    chest_pain = st.sidebar.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
    resting_bp = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 80, 200, 120)
    cholesterol = st.sidebar.slider('Serum Cholesterol (mg/dL)', 100, 600, 200)
    fasting_bs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dL', ['Yes', 'No'])
    rest_ecg = st.sidebar.selectbox('Resting ECG', ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy'])
    max_hr = st.sidebar.slider('Max Heart Rate Achieved', 60, 220, 150)
    exercise_angina = st.sidebar.selectbox('Exercise-Induced Angina', ['Yes', 'No'])
    oldpeak = st.sidebar.slider('Oldpeak (ST depression)', 0.0, 6.0, 1.0, step=0.1)
    st_slope = st.sidebar.selectbox('Slope of Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])

    sex_map = {'Male': 1, 'Female': 0}
    cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}
    fbs_map = {'Yes': 1, 'No': 0}
    ecg_map = {'Normal': 0, 'ST-T Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
    angina_map = {'Yes': 1, 'No': 0}
    slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}

    data = {
        'age': age,
        'sex': sex_map[sex],
        'chest_pain_type': cp_map[chest_pain],
        'resting_bp_s': resting_bp,
        'cholesterol': cholesterol,
        'fasting_blood_sugar': fbs_map[fasting_bs],
        'resting_ecg': ecg_map[rest_ecg],
        'max_heart_rate': max_hr,
        'exercise_angina': angina_map[exercise_angina],
        'oldpeak': oldpeak,
        'st_slope': slope_map[st_slope],
        'chol_age_ratio': cholesterol / age,
        'pulse_pressure': max_hr - resting_bp,
        'oldpeak_scaled': oldpeak / 6.0
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Feature scaling
scaler_cols = ['age', 'resting_bp_s', 'cholesterol', 'max_heart_rate', 'oldpeak', 'chol_age_ratio', 'pulse_pressure']
scaler = StandardScaler()
input_df[scaler_cols] = scaler.fit_transform(input_df[scaler_cols])

# Predict
prediction = model.predict(input_df)[0]
pred_prob = model.predict_proba(input_df)[0][1]

st.subheader("Prediction Result")
if prediction == 1:
    st.error(f"⚠️ High Risk of Heart Disease with {pred_prob * 100:.1f}% probability.")
    st.markdown("""
    #### What You Can Do:
    - 🥗 Improve your diet (low cholesterol, less salt)
    - 🏃‍♂️ Exercise regularly (30 mins/day)
    - 🚭 Quit smoking if applicable
    - 🧘‍♂️ Manage stress
    - 📞 Consult a cardiologist immediately
    """)
else:
    st.success(f"✅ Low Risk of Heart Disease with {pred_prob * 100:.1f}% probability.")
    st.markdown("""
    #### Keep it up!
    - 👍 Maintain healthy habits
    - 🩺 Do annual checkups
    - ❤️ Keep your heart strong!
    """)

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | By Ankush Kumar Jha")