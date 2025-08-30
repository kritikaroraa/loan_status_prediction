import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ---- Load saved objects ----
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

# ---- Streamlit App ----
st.title("🏦 Loan Status Prediction App")
st.write("Enter applicant details to predict loan approval status.")

# ---- User Input ----
def user_input():
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=0)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    
    data = {
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
    return pd.DataFrame([data])

input_df = user_input()

# ---- Preprocess Input ----
categorical_cols = ["Gender","Married","Dependents","Education","Self_Employed","Property_Area"]
numerical_cols = ["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]

# Encode categorical columns
for col in categorical_cols:
    le = label_encoders[col]
    # If unseen category, assign first known class
    input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
    input_df[col] = le.transform(input_df[col])

# Scale numeric columns
input_df[numerical_cols] = input_df[numerical_cols].astype(float)
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

# ---- Predict ----
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    result = "Approved ✅" if prediction == 1 else "Rejected ❌"
    st.success(f"Loan Status: {result}")
