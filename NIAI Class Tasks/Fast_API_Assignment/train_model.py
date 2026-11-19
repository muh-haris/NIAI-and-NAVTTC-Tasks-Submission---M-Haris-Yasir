"""
train_model.py  —  STARTER CODE (run this, don't modify it)
------------------------------------------------------------
Trains a K-Nearest Neighbours model on the REAL Pima Indians Diabetes
dataset (diabetes.csv, 768 patients) and saves it to model.pkl.

The model predicts whether a patient is likely DIABETIC from 8 inputs:

    Pregnancies              number of times pregnant
    Glucose                  plasma glucose concentration
    BloodPressure            diastolic blood pressure (mm Hg)
    SkinThickness            triceps skin fold thickness (mm)
    Insulin                  2-hour serum insulin (mu U/ml)
    BMI                      body mass index
    DiabetesPedigreeFunction family-history diabetes score
    Age                      age in years

    Outcome (label):  1 = Diabetic,  0 = Not Diabetic

You will build:
    - main.py          (FastAPI app that loads model.pkl and serves /predict)
    - streamlit_app.py (Streamlit UI that calls your /predict route)

Run it like this:
    pip install scikit-learn joblib numpy pandas
    python train_model.py
Keep diabetes.csv in the same folder as this file.
"""

import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── 1. Load the real dataset ─────────────────────────────────────────────────
df = pd.read_csv("diabetes.csv")

# These 8 columns are the INPUTS (features); "Outcome" is the ANSWER (label).
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
X = df[FEATURES].values
y = df["Outcome"].values

# ── 2. Split into train / test so we can measure honest accuracy ─────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3. Build the model ───────────────────────────────────────────────────────
# KNN measures DISTANCE between patients. These features have wildly different
# ranges (Glucose can be ~0–200, DiabetesPedigreeFunction ~0.08–2.4), so without
# scaling, Glucose and Insulin would dominate every distance. StandardScaler
# puts all 8 features on the same footing first.
#
# make_pipeline saves the scaler together with the classifier, so at predict
# time you pass RAW values and the scaling happens automatically inside.
model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)   # vote of the 5 nearest patients
)

model.fit(X_train, y_train)

# ── 4. Report accuracy on the held-out test set ──────────────────────────────
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Test accuracy: {acc:.2%}")

# ── 5. Save the whole pipeline to disk ───────────────────────────────────────
joblib.dump(model, "model.pkl")
print("Model trained and saved to model.pkl")

# ── 6. Example prediction (you'll reproduce this in main.py) ──────────────────
import numpy as np
sample = np.array([[2, 120, 70, 20, 79, 25.0, 0.5, 30]])  # one patient
pred = int(model.predict(sample)[0])
conf = round(float(model.predict_proba(sample)[0][pred]), 3)
print(f"Example -> prediction={pred} "
      f"({'Diabetic' if pred else 'Not Diabetic'})  confidence={conf}")
