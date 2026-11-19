"""Streamlit UI that calls FastAPI over HTTP; it never loads model.pkl."""

import os

import requests
import streamlit as st

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

# For local use, call FastAPI on port 8000.
# On Streamlit Community Cloud, set API_URL in the app's Secrets settings
# to the public HTTPS URL of your deployed FastAPI service.
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.title("🩺 Diabetes Risk Predictor")
st.write(
    "Enter the eight patient features below, then click **Predict**. "
    "This interface sends your inputs to the FastAPI /predict endpoint."
)

left, right = st.columns(2, gap="large")

with left:
    pregnancies = st.slider("Pregnancies", min_value=0, max_value=17, value=2)
    glucose = st.slider("Glucose", min_value=0, max_value=200, value=120)
    blood_pressure = st.slider(
        "BloodPressure (mm Hg)", min_value=0, max_value=130, value=70
    )
    skin_thickness = st.slider(
        "SkinThickness (mm)", min_value=0, max_value=100, value=20
    )

with right:
    insulin = st.slider("Insulin", min_value=0, max_value=850, value=79)
    bmi = st.slider("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    diabetes_pedigree = st.number_input(
        "DiabetesPedigreeFunction",
        min_value=0.0,
        max_value=2.5,
        value=0.5,
        step=0.001,
        format="%.3f",
    )
    age = st.slider("Age", min_value=18, max_value=90, value=30)

if st.button("Predict", type="primary", use_container_width=True):
    # The keys are the exact field names required by PatientFeatures.
    payload = {
        "Pregnancies": pregnancies,
        "Glucose": float(glucose),
        "BloodPressure": float(blood_pressure),
        "SkinThickness": float(skin_thickness),
        "Insulin": float(insulin),
        "BMI": float(bmi),
        "DiabetesPedigreeFunction": float(diabetes_pedigree),
        "Age": age,
    }

    try:
        with st.spinner("Requesting prediction from FastAPI..."):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            label = result["result_label"]
            confidence = float(result["confidence"])

        if label == "Diabetic":
            st.error("Prediction: Diabetic")
        elif label == "Not Diabetic":
            st.success("Prediction: Not Diabetic")
        else:
            raise ValueError("Unexpected prediction label")

        st.metric("Prediction confidence", f"{confidence:.1%}")

    except (requests.exceptions.RequestException, ValueError, KeyError, TypeError):
        st.warning(
            "Unable to get a prediction. Start the FastAPI server first "
            "(locally: `python -m uvicorn main:app --reload`), or check "
            "your deployed API_URL in Streamlit's settings."
        )

st.caption(
    "Educational demonstration only; this KNN model is not a medical diagnosis "
    "or a validated clinical risk score. Confidence reflects the model's "
    "nearest-neighbor vote, not a calibrated probability of diabetes."
)
