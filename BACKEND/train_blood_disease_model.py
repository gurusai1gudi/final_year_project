import os
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "clinical_blood_disease_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "FRONTEND", "blood_disease_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "FRONTEND", "blood_disease_encoder.pkl")

df = pd.read_csv(CSV_PATH)

print("Dataset shape:", df.shape)

FEATURES = [
    "Age", "Gender", "Temperature_C",
    "Pulse",
    "BP_Systolic", "BP_Diastolic",
    "SpO2",
    "Hemoglobin",
    "WBC",
    "Platelets",
    "RBC",
    "ESR",
    "CRP",
    "Dengue_NS1",
    "Malaria_Parasite",
    "Widal_Test"
]

TARGET = "Disease"

X = df[FEATURES].copy()
y = df[TARGET]
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].str.strip().str.lower()
binary_map = {"yes": 1, "no": 0, "positive": 1, "negative": 0}
gender_map = {"male": 0, "female": 1}

X["Gender"] = X["Gender"].map(gender_map)
X["Dengue_NS1"] = X["Dengue_NS1"].map(binary_map)
X["Malaria_Parasite"] = X["Malaria_Parasite"].map(binary_map)
X["Widal_Test"] = X["Widal_Test"].map(binary_map)

# Convert everything to numeric
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Diseases learned:", list(label_encoder.classes_))
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Model Accuracy: {acc * 100:.2f}%")
pickle.dump(model, open(MODEL_PATH, "wb"))
pickle.dump(label_encoder, open(ENCODER_PATH, "wb"))

print("✅ Blood Disease Model Saved")
print("📁", MODEL_PATH)
print("📁", ENCODER_PATH)
