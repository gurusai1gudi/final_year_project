import pandas as pd

DATASET_PATH = r"C:/Users/gudig/OneDrive/final_year_project/MEDICINE_RECOMMENDATION-main/enhanced_fever_medicine_recommendation.csv"

df = pd.read_csv(DATASET_PATH)

# Disease → Medicine mapping (ONLY FROM THIS FILE)
DISEASE_TO_MEDICINE = (
    df.groupby("Disease")["Recommended_Medication"]
    .apply(lambda x: list(set(x)))
    .to_dict()
)
