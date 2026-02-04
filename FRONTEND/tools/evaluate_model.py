import os
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FRONTEND = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET = os.path.abspath(os.path.join(FRONTEND, '..', 'DATASET'))

print('FRONTEND:', FRONTEND)
print('DATASET:', DATASET)

# Load Training.csv
train_path = os.path.join(DATASET, 'Training.csv')
print('Loading training data from', train_path)
df = pd.read_csv(train_path)

# Features: all columns except last 'prognosis'
feature_cols = list(df.columns[:-1])
X = df[feature_cols].astype(int).values

y_raw = df['prognosis']
le = LabelEncoder()
y = le.fit_transform(y_raw)

print('Classes (label -> disease):')
for i, c in enumerate(le.classes_):
    print(i, c)

# Train/test split (same as notebook: random_state=20, test_size=0.3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20, stratify=y)
print('Train/test sizes:', X_train.shape, X_test.shape)

# Load model (prefer native)
_native_candidates = [
    os.path.join(FRONTEND, 'xgboost.json'),
    os.path.join(FRONTEND, 'xgboost.model'),
    os.path.join(DATASET, 'xgboost.json'),
    os.path.join(DATASET, 'xgboost.model'),
]
_native = next((p for p in _native_candidates if os.path.exists(p)), None)
model = None

if _native:
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(_native)
        print('Loaded XGBClassifier native model:', _native)
    except Exception as e:
        print('Loaded Booster native model (will wrap predict). error=', e)
        booster = xgb.Booster()
        booster.load_model(_native)
        def predict_fn(X_in):
            dm = xgb.DMatrix(np.array(X_in))
            preds = booster.predict(dm)
            preds = np.array(preds)
            if preds.ndim > 1:
                return np.argmax(preds, axis=1)
            return (preds > 0.5).astype(int)
else:
    pkl_candidates = [os.path.join(FRONTEND, 'xgboost.pkl'), os.path.join(DATASET, 'xgboost.pkl')]
    pkl = next((p for p in pkl_candidates if os.path.exists(p)), pkl_candidates[0])
    print('Loading pickled model:', pkl)
    model = pickle.load(open(pkl, 'rb'))

# Predict
if model is not None:
    y_pred = model.predict(X_test)
else:
    y_pred = predict_fn(X_test)

acc = accuracy_score(y_test, y_pred)
print('\nAccuracy on test split: {:.4f}'.format(acc))

print('\nClassification report:')
print(classification_report(y_test, y_pred, target_names=le.classes_))

print('\nConfusion matrix:')
print(confusion_matrix(y_test, y_pred))
