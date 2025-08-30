import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.title("🏦 Loan Status Prediction App")
st.write("Enter applicant details to predict loan approval status.")

# Load models and encoders
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("target_encoder.pkl", "rb") as f:
    target_encoder = pickle.load(f)

# --- User Inputs ---
def user_input():
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.number_input("Dependents", 0, 10, 0)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", 0, 100000, 5000)
    coapplicant_income = st.number_input("Coapplicant Income", 0, 100000, 0)
    loan_amount = st.number_input("Loan Amount", 0, 100000, 10000)
    loan_amount_term = st.number_input("Loan Amount Term", 0, 500, 360)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    
    data = {
        "gender": gender,
        "married": married,
        "dependents": dependents,
        "education": education,
        "self_employed": self_employed,
        "applicant_income": applicant_income,
        "coapplicant_income": coapplicant_income,
        "loan_amount": loan_amount,
        "loan_amount_term": loan_amount_term,
        "credit_history": credit_history,
        "property_area": property_area
    }
    
    return pd.DataFrame([data])

input_df = user_input()

# --- Preprocess input ---
for col, le in label_encoders.items():
    if col in input_df.columns:
        input_df[col] = le.transform(input_df[col])

# Scale numerical features
numerical_cols = ["dependents","applicant_income","coapplicant_income","loan_amount","loan_amount_term"]
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

# --- Prediction ---
if st.button("Predict"):
    pred = model.predict(input_df)
    result = target_encoder.inverse_transform(pred)[0]
    st.success(f"Predicted Loan Status: {result}")
