# SmartMediCare — Disease Prediction & Drug Recommendation

Short README to get you running quickly and explain the project layout, how the ML model is used, and common maintenance tasks.

## Project overview
- A small Django web application that predicts a disease from a list of symptoms and returns description, precautions, medication and diet/workout suggestions.
- Frontend + server: `FRONTEND/` (Django project). Data: `DATASET/` (CSV files). Training/experiments: `BACKEND/` (Jupyter notebooks).
- ML: XGBoost classifier (primary) and LightGBM used during experimentation.

## Repo structure (important files)
- `FRONTEND/` — Django project root
	- `manage.py` — Django commands
	- `self/settings.py` — Django settings (DB location: `FRONTEND/db.sqlite3`)
	- `webapp/` — app code (views, models, templates)
	- `templates/` — HTML templates (`index.html`, `input.html`, `output.html`)
	- `static/` — site static assets (images, css, js)
	- `xgboost.json` (or `xgboost.pkl`) — model files (may be created by training scripts)
- `DATASET/` — CSV data used for training and lookups: `Training.csv`, `description.csv`, `precautions_df.csv`, `medications.csv`, `diets.csv`, `workout_df.csv`.
- `BACKEND/` — Jupyter notebooks used to train/evaluate models (`Drug_Recommendation.ipynb` etc.).
- `FRONTEND/tools/` — helper scripts added: `resave_xgb_model.py`, `test_predict.py`, `evaluate_model.py` etc.

## Quick prerequisites
- Python 3.8+ (the environment used here supports the packages listed in `Requirements.txt`).
- Key libs: Django, pandas, xgboost, scikit-learn, lightgbm (optional for training).

## Setup & run (PowerShell)
From the repository root (example shown for Windows PowerShell):

1) Create & activate virtual environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r Requirements.txt
```

2) Apply Django migrations and start dev server
```powershell
cd FRONTEND
python manage.py migrate
python manage.py runserver
# open http://127.0.0.1:8000/ in your browser
```

3) (Optional) Create a superuser for admin
```powershell
python manage.py createsuperuser
```

## How inference works (high level)
- The user submits symptoms on the input page. The view builds a binary feature vector (symptom present = 1) in the exact order expected by the trained model and calls the XGBoost model's `predict` method.
- To avoid brittle mismatches, the code now attempts to read the `Training.csv` header and LabelEncoder classes at runtime so that feature ordering and label mapping match training.
- Model files: prefer native XGBoost model (`FRONTEND/xgboost.json` or `.model`). If not present, code falls back to `xgboost.pkl` (pickle).

## Evaluate model performance (quick)
I added a small evaluation script that reproduces a train/test split and reports accuracy, classification report and confusion matrix.

Run it from the repo root:
```powershell
python FRONTEND\tools\evaluate_model.py
```
It prints the classes, test split sizes, accuracy, and a full classification report.

Notes about reported perfect accuracy: the sample dataset and split used here produced 100% accuracy on the held-out split. This can indicate:
- a very clean / synthetic dataset,
- duplicates or leakage between training and test sets, or
- that the task is straightforward for the current feature representation.

For proper evaluation use a separate holdout set before hyperparameter tuning, or run cross-validation and inspect confusion matrices / per-class performance.

## Improve accuracy — practical steps
- Persist the training pipeline: save feature order and LabelEncoder after training (joblib/pickle). Load these at inference to guarantee identical preprocessing.
- Investigate class balance in `Training.csv`. Use stratified CV, class weights, or oversampling (SMOTE) if classes are imbalanced.
- Hyperparameter tuning (GridSearchCV, RandomizedSearchCV, or Optuna) for XGBoost / LightGBM.
- Inspect feature importance (XGBoost feature_importances_ or SHAP) to remove noisy features.
- Consider ensembling XGBoost and LightGBM (voting or stacking).

## Useful developer scripts
- `FRONTEND/tools/resave_xgb_model.py` — helper to convert a pickled XGBoost model to native `save_model()` JSON/binary.
- `FRONTEND/tools/test_predict.py` — quick local smoke-test loader that builds a sample feature vector and predicts.
- `FRONTEND/tools/evaluate_model.py` — runs a train/test split and prints performance metrics.

## Quick GitHub push (example commands)
If you want to push this repo to GitHub, first add a `.gitignore` (recommended) and then:

Using GitHub CLI (recommended):
```powershell
gh auth login
gh repo create <your-username>/MEDICINE_RECOMMENDATION --public --source=. --remote=origin --push
```

Manual via web + git:
```powershell
git init            # if not already a git repo
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/MEDICINE_RECOMMENDATION.git
git push -u origin main
```

Notes: if you use HTTPS and have 2FA, use a personal access token (PAT) as the password; SSH keys are another option.

## Security & deployment notes
- Do NOT commit `FRONTEND/self/settings.py` secrets into a public repo. Move `SECRET_KEY` to an environment variable for production and set `DEBUG=False`.
- Use a production-ready DB (Postgres) and configure static/media file serving and HTTPS for production.

## Troubleshooting
- If the app returns unexpected predictions, run `FRONTEND/tools/test_predict.py` to confirm the saved model predicts expected labels from a known vector.
- Run `python FRONTEND/tools/evaluate_model.py` to reproduce accuracy numbers and inspect the confusion matrix.
- If you see XGBoost pickle compatibility warnings when loading `.pkl`, re-save with `save_model()` (see `resave_xgb_model.py`).

## Contact / next steps
- For review prep I can: add pipeline serialization to the notebook, run CV and hyperparameter tuning, wire per-disease images into `output.html`, or prepare a short demo script. Tell me which one you'd like next.

---
Generated/updated: January 2026

