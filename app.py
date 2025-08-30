import pandas as pd
import streamlit as st
import pickle

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
    education = st.selectbox("Education", ["High School or Below", "Bechalor", "college", "Master or Above"])
    principal = st.number_input("Loan Principal Amount", min_value=0)
    terms = st.number_input("Loan Terms (in days)", min_value=0)
    age = st.number_input("Applicant Age", min_value=18, max_value=100)

    data = {
        "Gender": gender,
        "education": education,
        "Principal": principal,
        "terms": terms,
        "age": age
    }
    return pd.DataFrame([data])

input_df = user_input()

# ---- Preprocess Input ----
categorical_cols = ["Gender", "education"]
numerical_cols = ["Principal", "terms", "age"]

# Encode categorical columns safely
for col in categorical_cols:
    if col not in label_encoders:
        st.error(f"⚠️ No encoder found for column: {col}")
        continue
    le = label_encoders[col]
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
