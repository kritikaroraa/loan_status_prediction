# Loan Prediction ML Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Load dataset
df = pd.read_csv("loan_dataset.csv")

# Quick look at data
print(df.head())
print(df.info())

# 2. Handle missing values
df = df.dropna()

# 3. Encode categorical columns
categorical_cols = ["Gender", "education"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target variable
target_encoder = LabelEncoder()
df["loan_status"] = target_encoder.fit_transform(df["loan_status"])

# 4. Split features & target
X = df[["Gender", "education", "Principal", "terms", "age"]]
y = df["loan_status"]

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Scale numeric columns
numerical_cols = ["Principal", "terms", "age"]
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

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

# ---- Save trained model, scaler, encoders ----
with open("loan_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("target_encoder.pkl", "wb") as f:
    pickle.dump(target_encoder, f)

print("✅ Models and encoders saved successfully!")
