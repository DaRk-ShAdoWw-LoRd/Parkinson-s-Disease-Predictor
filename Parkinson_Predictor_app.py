import streamlit as st
import pandas as pd
import joblib

# Page Config
st.set_page_config(
    page_title="Parkinson's Disease Predictor",
    layout="centered"
)

# Load trained pipeline
model = joblib.load("Parkinson_Predictor_Model.pkl")

# Title
st.title("Parkinson's Disease Predictor")

st.markdown("""
This application predicts the likelihood of Parkinson's disease
using voice measurement features and an ensemble machine learning model.
""")

# Sidebar
st.sidebar.header("About")

st.sidebar.info(
    """
    This predictor uses a Soft Voting Ensemble consisting of:
    
    • SVC  
    • XGBoost  
    • Random Forest  
    
    The model was trained on Parkinson's voice measurement data.
    """
)

st.sidebar.warning(
    "This tool is for educational purposes only and is not a substitute for professional medical diagnosis."
)

# Example Buttons
st.subheader("Load Example Values")

col1, col2 = st.columns(2)

healthy_example = {
    "Average Fundamental Frequency": 220.00,
    "Main Jitter Pct Variation": 0.00,
    "Jitter Microseconds": 0.00,
    "Relative Average Perturbation": 0.00,
    "Core Shimmer Amplitude Ratio": 0.01,
    "3 Cycle Smoothed Shimmer": 0.01,
    "Harmonics to Noise Ratio": 32.00,
    "Recurrence Period Density Entropy": 0.30,
    "Detrended Fluctuation Analysis": 0.60,
    "Nonlinear Freq Variation 1": -7.50,
    "Nonlinear Freq Variation 2": 0.10,
    "Correlation Dimension": 1.50,
    "Pitch Period Entropy": 0.05
}

parkinsons_example = {
    "Average Fundamental Frequency": 120.00,
    "Main Jitter Pct Variation": 0.02,
    "Jitter Microseconds": 0.00,
    "Relative Average Perturbation": 0.01,
    "Core Shimmer Amplitude Ratio": 0.07,
    "3 Cycle Smoothed Shimmer": 0.05,
    "Harmonics to Noise Ratio": 10.00,
    "Recurrence Period Density Entropy": 0.70,
    "Detrended Fluctuation Analysis": 0.80,
    "Nonlinear Freq Variation 1": -3.00,
    "Nonlinear Freq Variation 2": 0.40,
    "Correlation Dimension": 3.00,
    "Pitch Period Entropy": 0.40
}

if "inputs" not in st.session_state:
    st.session_state.inputs = healthy_example.copy()

with col1:
    if st.button("Load Healthy Example"):
        st.session_state.inputs = healthy_example.copy()

with col2:
    if st.button("Load Parkinson's Example"):
        st.session_state.inputs = parkinsons_example.copy()

st.subheader("Enter Voice Measurement Features")

# Input fields
inputs = {}

for feature, default in st.session_state.inputs.items():

    inputs[feature] = st.number_input(
        feature,
        value=float(default),
        format="%.2f"
    )

# Prediction Button
if st.button("Predict"):

    data = pd.DataFrame([inputs])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    st.subheader("Prediction Result")

    st.progress(float(probability))

    if prediction == 1:

        st.error(
            f"Parkinson's Detected\n\nConfidence: {probability:.2%}"
        )

    else:

        st.success(
            f"No Parkinson's Detected\n\nConfidence: {(1 - probability):.2%}"
        )

    st.subheader("Input Summary")
    st.dataframe(data)

st.caption(
    "Educational use only. Consult qualified medical professionals for diagnosis."
)
