import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "enhanced_fever_medicine_recommendation.csv")

df = pd.read_csv(CSV_PATH)

FEATURES = [
    "Age", "Gender", "BMI",
    "Smoking_History", "Alcohol_Consumption",
    "Headache", "Body_Ache", "Fatigue",
    "Fever_Severity"
]

X = df[FEATURES].copy()

# normalize strings
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].str.strip().str.lower()

binary = {"yes": 1, "no": 0}
gender = {"male": 0, "female": 1}
severity = {"low": 1, "medium": 2, "high": 3}

X["Gender"] = X["Gender"].map(gender)
X["Smoking_History"] = X["Smoking_History"].map(binary)
X["Alcohol_Consumption"] = X["Alcohol_Consumption"].map(binary)
X["Headache"] = X["Headache"].map(binary)
X["Body_Ache"] = X["Body_Ache"].map(binary)
X["Fatigue"] = X["Fatigue"].map(binary)
X["Fever_Severity"] = X["Fever_Severity"].map(severity)

X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

print("Final feature dtypes:\n", X.dtypes)

le = LabelEncoder()
df["combined_med"] = (
    df["Previous_Medication"].astype(str)
    + " -> "
    + df["Recommended_Medication"].astype(str)
)

y = le.fit_transform(df["combined_med"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(
    max_depth=4,
    n_estimators=120,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

pickle.dump(model, open(os.path.join(BASE_DIR, "FRONTEND/fever_model.pkl"), "wb"))
pickle.dump(le, open(os.path.join(BASE_DIR, "FRONTEND/medicine_encoder.pkl"), "wb"))

print("✅ Fever → Medicine model trained successfully")
