import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Loan Status Prediction", page_icon="🏦")
st.title("🏦 Loan Status Prediction App")
st.write("Enter applicant details to predict loan approval status.")

# --- Load model and preprocessing objects ---
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("target_encoder.pkl", "rb") as f:
    target_encoder = pickle.load(f)

# --- Function to get user input ---
def get_user_input():
    gender = st.selectbox("Gender", ["Male", "Female"]).title()
    married = st.selectbox("Married", ["Yes", "No"]).title()
    dependents = st.number_input("Dependents", 0, 10, 0)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"]).title()
    self_employed = st.selectbox("Self Employed", ["Yes", "No"]).title()
    applicant_income = st.number_input("Applicant Income", 0, 100000, 5000)
    coapplicant_income = st.number_input("Coapplicant Income", 0, 100000, 0)
    loan_amount = st.number_input("Loan Amount", 0, 100000, 10000)
    loan_amount_term = st.number_input("Loan Amount Term", 0, 500, 360)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"]).title()
    
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

input_df = get_user_input()

# --- Preprocess input safely ---
for col, le in label_encoders.items():
    if col in input_df.columns:
        # Replace unseen categories with most frequent training class
        input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        input_df[col] = le.transform(input_df[col])

# --- Scale numerical features ---
numerical_cols = ['ApplicantIncome','CoapplicantIncome','Dependents','LoanAmount','Loan_Amount_Term']
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

# --- Display processed input (optional) ---
st.subheader("Processed Input Data")
st.write(input_df)

# --- Predict ---
if st.button("Predict Loan Status"):
    prediction = model.predict(input_df)
    result = target_encoder.inverse_transform(prediction)[0]
    st.success(f"Predicted Loan Status: {result}")
