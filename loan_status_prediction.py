# Loan Prediction ML Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

# 1. Load dataset
df = pd.read_csv("loan_dataset.csv")

# Quick look at data
print(df.head())
print(df.info())

# 2. Handle missing values (if any)
df = df.dropna()

# 3. Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    if col != "loan_status":   # Don't encode target here
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Encode target variable
target_encoder = LabelEncoder()
df["loan_status"] = target_encoder.fit_transform(df["loan_status"])

# 4. Split features & target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Train Model (Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 8. Predictions
y_pred = model.predict(X_test)

# 9. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 10. Feature importance
importances = model.feature_importances_
feat_importances = pd.Series(importances, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.show()

import pickle

# Save trained Random Forest model
with open("loan_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the scaler too (important!)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save label encoders
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("target_encoder.pkl", "wb") as f:
    pickle.dump(target_encoder, f)

print("Models and encoders saved!")
