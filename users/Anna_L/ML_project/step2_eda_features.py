# STEP 2 — EDA & FEATURE ENGINEERING
# Analyse variance, correlations, outliers.
# Create interaction terms and thematic group features.
# Saves: data/flood_data_featured.pkl

import pandas as pd
import numpy as np
import pickle

# Load raw data from step 1
with open("data/flood_data_raw.pkl", "rb") as f:
    df = pickle.load(f)

print("=" * 60)
print("STEP 2 — EDA & FEATURE ENGINEERING")
print("=" * 60)

features = [c for c in df.columns if c != "FloodProbability"]

# 2a: Target distribution
print("\n[2a] Target — FloodProbability statistics:")
print(df["FloodProbability"].describe().round(4).to_string())
print("\n  NOTE: FloodProbability = mean(features) / 10 (synthetic dataset).")
print("        TotalRiskScore is therefore EXCLUDED — it would be data leakage.")

# 2b: Feature variance
# Low variance → low information content → consider dropping
print("\n[2b] Feature variance (ascending):")
variance = df[features].var().sort_values()
print(variance.round(3).to_string())

low_var = variance[variance < 0.01]
if len(low_var) > 0:
    print(f"\n  Low-variance features (consider dropping): {low_var.index.tolist()}")
else:
    print("\n  No low-variance features found — keeping all columns.")

# 2c: Pearson correlation with target
# Pearson: measures linear relationship between each feature and FloodProbability
# Range: -1 (perfect negative) to +1 (perfect positive), 0 = no linear relation
print("\n[2c] Pearson correlation with FloodProbability:")
corr_target = df[features].corrwith(df["FloodProbability"]).sort_values(ascending=False)
print(corr_target.round(4).to_string())
mean_corr = corr_target.mean().round(2)
print(f"\n  All features correlate equally (~{mean_corr}) — each contributes the same weight.")

# 2d: Outlier detection (IQR method)
# A value is an outlier if it lies below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
print("\n[2d] Outlier detection (IQR method):")
found_outliers = False
for col in features:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    n = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
    if n > 0:
        print(f"  {col}: {n} outliers")
        found_outliers = True
if not found_outliers:
    print("  No outliers found — uniform 1-10 scale is clean.")

# 2e: Feature engineering
# NOTE: TotalRiskScore (sum of all features) is NOT created — data leakage.
# Instead we create:
#   - Interaction terms: product of two related features
#   - Thematic group means: average of features sharing a common theme

print("\n[2e] Feature engineering:")

df["Monsoon_x_Topo"]   = df["MonsoonIntensity"]  * df["TopographyDrainage"]
df["Monsoon_x_River"]  = df["MonsoonIntensity"]  * df["RiverManagement"]
df["Urban_x_Drainage"] = df["Urbanization"]      * df["DrainageSystems"]
df["Deforest_x_Silt"]  = df["Deforestation"]     * df["Siltation"]

df["NaturalRisk"] = df[[
    "MonsoonIntensity", "TopographyDrainage",
    "Landslides", "Watersheds", "CoastalVulnerability"
]].mean(axis=1)

df["HumanRisk"] = df[[
    "Deforestation", "Urbanization", "Encroachments",
    "AgriculturalPractices", "WetlandLoss",
    "InadequatePlanning", "PoliticalFactors", "PopulationScore"
]].mean(axis=1)

df["InfraRisk"] = df[[
    "RiverManagement", "DamsQuality", "DrainageSystems",
    "DeterioratingInfrastructure", "IneffectiveDisasterPreparedness"
]].mean(axis=1)

all_features = [c for c in df.columns if c != "FloodProbability"]

print(f"  Interaction terms : Monsoon_x_Topo, Monsoon_x_River, Urban_x_Drainage, Deforest_x_Silt")
print(f"  Group means       : NaturalRisk, HumanRisk, InfraRisk")
print(f"  Dataset shape after engineering: {df.shape}")
print(f"  Total features for model: {len(all_features)}")

# Save engineered dataframe
with open("data/flood_data_featured.pkl", "wb") as f:
    pickle.dump({"df": df, "all_features": all_features}, f)

print("\nEngineered dataframe saved to: data/flood_data_featured.pkl")
print("=" * 60)
print("STEP 2 COMPLETE")
print("=" * 60)