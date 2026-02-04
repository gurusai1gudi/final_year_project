from django.db import models # type: ignore
import numpy as np
import pickle
import xgboost as xgb

# Load the trained ML model (replace with the actual model path)
import os
from django.conf import settings

# Resolve model path relative to project structure
# settings.BASE_DIR points to the FRONTEND folder; DATASET is a sibling of FRONTEND
PROJECT_ROOT = os.path.abspath(os.path.join(settings.BASE_DIR, '..'))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'DATASET')
model_path_candidates = [
    os.path.join(DATASET_DIR, 'xgboost.json'),
    os.path.join(DATASET_DIR, 'xgboost.model'),
    os.path.join(DATASET_DIR, 'xgboost.pkl'),
    os.path.join(settings.BASE_DIR, 'xgboost.json'),
    os.path.join(settings.BASE_DIR, 'xgboost.model'),
    os.path.join(settings.BASE_DIR, 'xgboost.pkl'),
]
model_path = next((p for p in model_path_candidates if os.path.exists(p)), None)

if model_path and model_path.endswith(('.json', '.model')):
    # Load native model
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(model_path)
    except Exception:
        booster = None
        try:
            booster = xgb.Booster()
            booster.load_model(model_path)
        except Exception:
            raise RuntimeError(f"Found native model at {model_path} but failed to load it")

        class _BoosterWrapper:
            def __init__(self, booster):
                self.booster = booster
            def predict(self, X):
                dm = xgb.DMatrix(np.array(X))
                return self.booster.predict(dm)

        model = _BoosterWrapper(booster)
elif model_path:
    # fallback to pickle
    model = pickle.load(open(model_path, 'rb'))
else:
    raise FileNotFoundError('No model file found in expected locations')

# Function to predict disease based on symptoms
def predict_disease(symptom_list):
    """
    Predict the disease based on the given symptoms.

    Args:
        symptom_list (list): A list of symptom values (binary: 1 if present, 0 if not).

    Returns:
        str: Predicted disease name.
    """
    symptoms_array = np.array(symptom_list).reshape(1, -1)  # Reshape for model input
    predicted_disease = model.predict(symptoms_array)[0]  # Get the prediction
    return predicted_disease
