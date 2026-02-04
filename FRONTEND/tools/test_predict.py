import os
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle

FRONTEND = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET = os.path.abspath(os.path.join(FRONTEND, '..', 'DATASET'))

# find native model
_native_candidates = [
    os.path.join(FRONTEND, 'xgboost.json'),
    os.path.join(FRONTEND, 'xgboost.model'),
    os.path.join(DATASET, 'xgboost.json'),
    os.path.join(DATASET, 'xgboost.model'),
]
_native = next((p for p in _native_candidates if os.path.exists(p)), None)

model = None
predict_fn = None
if _native:
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(_native)
        print('Loaded XGBClassifier native model:', _native)
    except Exception as e:
        print('Loaded Booster native model, wrapping predict. err=', e)
        booster = xgb.Booster()
        booster.load_model(_native)
        def predict_fn(X):
            dm = xgb.DMatrix(np.array(X))
            preds = booster.predict(dm)
            preds = np.array(preds)
            if preds.ndim > 1:
                return np.argmax(preds, axis=1)
            return (preds > 0.5).astype(int)
else:
    _pkl_candidates = [os.path.join(FRONTEND, 'xgboost.pkl'), os.path.join(DATASET, 'xgboost.pkl')]
    _pkl = next((p for p in _pkl_candidates if os.path.exists(p)), _pkl_candidates[0])
    model = pickle.load(open(_pkl, 'rb'))
    print('Loaded pickled model:', _pkl)

# build normalized feature list
_train_df = pd.read_csv(os.path.join(DATASET, 'Training.csv'), nrows=0)
feature_cols = list(_train_df.columns[:-1])
normalized = [c.strip().replace(' ', '_') for c in feature_cols]
print('Feature count:', len(normalized))

# create a sample vector for a known case (Fungal infection: itching, skin_rash, nodal_skin_eruptions)
vec = np.zeros(len(normalized))
sample_symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions']
for s in sample_symptoms:
    if s in normalized:
        vec[normalized.index(s)] = 1

if model is not None:
    pred = model.predict([vec])[0]
else:
    pred = predict_fn([vec])[0]
print('Predicted label (int):', pred)

# show label->disease mapping
from sklearn.preprocessing import LabelEncoder
_train = pd.read_csv(os.path.join(DATASET, 'Training.csv'), usecols=['prognosis'])
le = LabelEncoder(); le.fit(_train['prognosis'])
print('Label mapping (index: disease):')
for i,c in enumerate(le.classes_):
    print(i, c)

print('Predicted disease name:', le.inverse_transform([int(pred)])[0])
