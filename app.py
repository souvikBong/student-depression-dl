import streamlit as st
import torch
import pandas as pd
import pickle
import os

from src.model.model import DepressionModel
from src.data.preprocess import preprocess_data

# -----------------------------
# 1. Base Path Setup
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# 2. Load Feature Columns
# -----------------------------
with open(os.path.join(BASE_DIR, "feature_columns.pkl"), "rb") as f:
    feature_columns = pickle.load(f)

# -----------------------------
# 3. Load Model
# -----------------------------
input_size = len(feature_columns)

model = DepressionModel(input_size)
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "model.pth")))
model.eval()

# -----------------------------
# 4. Streamlit UI
# -----------------------------
st.title("🎓 Student Depression Prediction")

st.write("Enter the details below to assess depression risk:")

age = st.number_input("Age", 15, 40)
cgpa = st.number_input("CGPA", 0.0, 10.0)
study_hours = st.number_input("Work/Study Hours", 0, 12)

sleep = st.selectbox("Sleep Duration", [
    "Less than 5 hours",
    "5-6 hours",
    "7-8 hours",
    "More than 8 hours"
])

diet = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])

family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
suicidal = st.selectbox("Have you ever had suicidal thoughts ?", ["Yes", "No"])

academic_pressure = st.slider("Academic Pressure", 1, 5)
financial_stress = st.slider("Financial Stress", 1, 5)

# -----------------------------
# 5. Prediction Logic
# -----------------------------
if st.button("Predict"):

    # Create input dataframe
    data = {
        'Age': age,
        'CGPA': cgpa,
        'Work/Study Hours': study_hours,
        'Sleep Duration': sleep,
        'Dietary Habits': diet,
        'Family History of Mental Illness': family_history,
        'Have you ever had suicidal thoughts ?': suicidal,
        'Academic Pressure': academic_pressure,
        'Financial Stress': financial_stress,
        'Gender': 'Male',
        'Study Satisfaction': 3,
        'Job Satisfaction': 3,
        'Work Pressure': 3,
        'Depression': 0  # dummy column for preprocessing
    }

    df = pd.DataFrame([data])

    # -----------------------------
    # 6. Preprocess
    # -----------------------------
    df_processed = preprocess_data(df)

    X = df_processed.drop(columns=['Depression'])

    # 🔥 CRITICAL: Match training features
    X = X.reindex(columns=feature_columns, fill_value=0)

    # -----------------------------
    # 7. Convert to Tensor
    # -----------------------------
    X_tensor = torch.tensor(X.values, dtype=torch.float32)

    # -----------------------------
    # 8. Prediction
    # -----------------------------
    with torch.no_grad():
        output = model(X_tensor)
        prob = output.item()

    prediction = 1 if prob > 0.5 else 0

    # -----------------------------
    # 9. Display Result
    # -----------------------------
    if prediction == 1:
        st.error(f"⚠️ High Risk of Depression ({prob*100:.2f}% confidence)")
    else:
        st.success(f"✅ Low Risk of Depression ({(1-prob)*100:.2f}% confidence)")

    # Optional debug info (can remove later)
    st.write("Feature vector shape:", X.shape)