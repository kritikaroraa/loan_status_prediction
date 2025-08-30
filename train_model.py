import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# ---- Load dataset ----
df = pd.read_csv("loan_dataset.csv")
df = df.dropna()

# ---- Encode categorical columns ----
categorical_cols = ["Gender", "education"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ---- Encode target ----
target_encoder = LabelEncoder()
df["loan_status"] = target_encoder.fit_transform(df["loan_status"])

# ---- Features & Target ----
X = df[["Gender", "education", "Principal", "terms", "age"]]
y = df["loan_status"]

# ---- Scale numerical columns ----
numerical_cols = ["Principal", "terms", "age"]
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# ---- Train model ----
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# ---- Save objects ----
pickle.dump(model, open("loan_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))
pickle.dump(target_encoder, open("target_encoder.pkl", "wb"))

print("✅ Model, scaler, and encoders saved successfully!")
