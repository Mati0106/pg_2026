# STEP 1 — DATA COLLECTION
# Load the dataset, inspect basic properties, check for missing values.
# Saves: data/flood_data_raw.pkl

import pandas as pd
import pickle
import os

os.makedirs("data", exist_ok=True)

# Load
df = pd.read_csv("flood_data.csv")

# Basic inspection
print("=" * 60)
print("STEP 1 — DATA LOADING")
print("=" * 60)

print(f"\nShape : {df.shape[0]} rows, {df.shape[1]} columns")

print(f"\nFirst rows:")
print(df.head(3).to_string())

print(f"\nData types:")
print(df.dtypes.to_string())

print(f"\nMissing values per column:")
missing = df.isnull().sum()
print(missing[missing > 0].to_string() if missing.sum() > 0 else "  None — dataset is clean.")

print(f"\nTarget column basic stats (FloodProbability):")
print(df["FloodProbability"].describe().round(4).to_string())

# NOTE on dataset
# FloodProbability = mean(all feature columns) / 10  → synthetic dataset.
# Consequence: TotalRiskScore (sum of features) is EXCLUDED from feature
# engineering to avoid data leakage (model would trivially recover the target).

# Save raw dataframe
with open("data/flood_data_raw.pkl", "wb") as f:
    pickle.dump(df, f)

print("\nRaw dataframe saved to: data/flood_data_raw.pkl")
print("=" * 60)
print("STEP 1 COMPLETE")
print("=" * 60)