# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# -------------------------------
# Load trained model & preprocessors
# -------------------------------
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# -------------------------------
# Streamlit App
# -------------------------------
st.title("🏦 Loan Status Prediction App")
st.write("Enter applicant details to predict loan approval status.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term", min_value=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Prepare input dataframe
input_dict = {
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_amount_term,
    "Credit_History": credit_history,
    "Property_Area": property_area
}

input_df = pd.DataFrame([input_dict])

# -------------------------------
# Encode categorical columns
# -------------------------------
for col, le in label_encoders.items():
    if col in input_df.columns:
        # handle unseen labels
        input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        input_df[col] = le.transform(input_df[col])

# -------------------------------
# Scale numeric columns
# -------------------------------
numerical_cols = ["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

# -------------------------------
# Prediction
# -------------------------------
prediction = model.predict(input_df)[0]

status = "Approved ✅" if prediction == 1 else "Rejected ❌"
st.success(f"The loan is likely to be: {status}")
