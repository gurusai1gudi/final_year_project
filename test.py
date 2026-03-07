import pandas as pd

df = pd.read_csv("clinical_blood_disease_dataset.csv")
unique_diseases=[]
for disease in df["Disease"]:
    if disease not in unique_diseases:
        unique_diseases.append(disease)
print(unique_diseases)
