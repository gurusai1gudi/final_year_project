import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# Paths

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "enhanced_fever_with_fever_type.csv")


# Load dataset

df = pd.read_csv(CSV_PATH)


# Features used for training

FEATURES = [
    "Age",
    "Gender",
    "BMI",
    "Smoking_History",
    "Alcohol_Consumption",
    "Headache",
    "Body_Ache",
    "Fatigue",
    "Fever_Severity"
]

X = df[FEATURES].copy()

# Normalize text columns

for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].astype(str).str.strip().str.lower()


# Encoding maps

binary = {"yes": 1, "no": 0}
gender = {"male": 0, "female": 1}
severity = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "normal": 1,
    "mild fever": 2,
    "high fever": 3
}

X["Gender"] = X["Gender"].map(gender)
X["Smoking_History"] = X["Smoking_History"].map(binary)
X["Alcohol_Consumption"] = X["Alcohol_Consumption"].map(binary)
X["Headache"] = X["Headache"].map(binary)
X["Body_Ache"] = X["Body_Ache"].map(binary)
X["Fatigue"] = X["Fatigue"].map(binary)
X["Fever_Severity"] = X["Fever_Severity"].map(severity)

# Convert everything to numeric
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

print(" Feature columns ready")
print(X.head())


# Target: Fever_Type

le = LabelEncoder()
y = le.fit_transform(df["Fever_Type"].astype(str))

print(" Fever types:", list(le.classes_))


# Train / Test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Model

model = XGBClassifier(
    max_depth=4,
    n_estimators=120,
    learning_rate=0.1,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)


# Save model & encoder

pickle.dump(
    model,
    open(os.path.join(BASE_DIR, "FRONTEND", "fever_type_model.pkl"), "wb")
)

pickle.dump(
    le,
    open(os.path.join(BASE_DIR, "FRONTEND", "fever_type_encoder.pkl"), "wb")
)

print(" Fever Type model trained successfully")
print(" Saved files:")
print(" - fever_type_model.pkl")
print(" - fever_type_encoder.pkl")
