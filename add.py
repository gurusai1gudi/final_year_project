import pandas as pd
import random
import os

# Project root (your confirmed path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE_DIR, "enhanced_fever_medicine_recommendation.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "enhanced_fever_with_fever_type.csv")

# Load CSV
df = pd.read_csv(CSV_PATH)

def assign_fever_type(severity):
    severity = str(severity).strip().lower()

    if severity == "high fever":
        return random.choice(["Typhoid", "Dengue", "Malaria"])
    elif severity == "mild fever":
        return random.choice(["Viral Fever", "Infection Fever"])
    elif severity == "normal":
        return "Normal fever due to stress"
    else:
        return "Unknown"

# Add new column
df["Fever_Type"] = df["Fever_Severity"].apply(assign_fever_type)

# Save new CSV
df.to_csv(OUTPUT_PATH, index=False)

print("✅ Fever_Type column added successfully")
print("📁 Output file:", OUTPUT_PATH)




