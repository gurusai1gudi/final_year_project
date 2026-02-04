import os
import pickle
import xgboost as xgb

# Compute paths relative to this script file so the script works when run from any CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../FRONTEND/tools
FRONTEND_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(FRONTEND_DIR, '..'))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'DATASET')

# Candidate pickle filenames (common names used in this project)
candidate_names = ['xgboost.pkl', 'xgb_model.pkl', 'xgb.pkl']
PICKLE_CANDIDATES = [os.path.join(FRONTEND_DIR, n) for n in candidate_names] + [os.path.join(DATASET_DIR, n) for n in candidate_names]

PICKLE_PATH = next((p for p in PICKLE_CANDIDATES if os.path.exists(p)), None)
OUT_JSON = os.path.join(FRONTEND_DIR, 'xgboost.json')   # native json format (output)
OUT_MODEL = os.path.join(FRONTEND_DIR, 'xgboost.model') # native binary format (output)

if not PICKLE_PATH:
    raise SystemExit(f"Pickle file not found. Tried:\n" + "\n".join(PICKLE_CANDIDATES))

print("Loading pickled model (may print warnings)... ->", PICKLE_PATH)
with open(PICKLE_PATH, 'rb') as f:
    model = pickle.load(f)

# Try direct save (sklearn wrapper has .save_model, or use get_booster())
saved = False
try:
    # XGBClassifier / XGBRegressor wrapper exposes save_model in recent versions
    model.save_model(OUT_JSON)
    print("Saved model using model.save_model() ->", OUT_JSON)
    saved = True
except Exception:
    try:
        # If it's an sklearn wrapper, get the Booster and save it
        booster = model.get_booster()
        booster.save_model(OUT_JSON)
        print("Saved booster via get_booster() ->", OUT_JSON)
        saved = True
    except Exception:
        try:
            # If the object is actually a Booster
            if isinstance(model, xgb.Booster):
                model.save_model(OUT_JSON)
                print("Saved Booster ->", OUT_JSON)
                saved = True
        except Exception:
            pass

if not saved:
    raise RuntimeError("Could not save model in native XGBoost format. Inspect the `model` object type and attributes.")