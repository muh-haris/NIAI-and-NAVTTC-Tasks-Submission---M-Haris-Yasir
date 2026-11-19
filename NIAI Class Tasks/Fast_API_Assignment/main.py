"""FastAPI prediction service for the provided diabetes KNN model."""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load the saved model only at startup, not for each prediction request.
ml_model = {}
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model["predictor"] = joblib.load(MODEL_PATH)
    try:
        yield
    finally:
        ml_model.clear()


app = FastAPI(title="Diabetes Risk API", lifespan=lifespan)


class PatientFeatures(BaseModel):
    """The eight inputs, in exactly the order used by train_model.py."""

    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


class PredictionResult(BaseModel):
    prediction: int
    result_label: str
    confidence: float


@app.get("/")
def health():
    return {"status": "running", "model_loaded": "predictor" in ml_model}


@app.post("/predict", response_model=PredictionResult)
def predict(features: PatientFeatures):
    model = ml_model.get("predictor")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Feature order must match the dataset columns used in training.
    X = np.array(
        [[
            features.Pregnancies,
            features.Glucose,
            features.BloodPressure,
            features.SkinThickness,
            features.Insulin,
            features.BMI,
            features.DiabetesPedigreeFunction,
            features.Age,
        ]],
        dtype=float,
    )

    pred = int(model.predict(X)[0])
    conf = round(float(model.predict_proba(X)[0][pred]), 3)

    return PredictionResult(
        prediction=pred,
        result_label="Diabetic" if pred == 1 else "Not Diabetic",
        confidence=conf,
    )
