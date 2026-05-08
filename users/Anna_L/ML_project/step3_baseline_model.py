# STEP 3 — BASELINE MODELING
# Split data 80/20, train XGBoost with default parameters as a baseline.
# The baseline gives us a reference point before hyperparameter tuning.
# Saves: data/split.pkl, data/baseline_scores.pkl

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load engineered data from step 2
with open("data/flood_data_featured.pkl", "rb") as f:
    payload = pickle.load(f)

df           = payload["df"]
all_features = payload["all_features"]

print("=" * 60)
print("STEP 3 — BASELINE MODELING")
print("=" * 60)

# Train / test split (80 / 20)
# random_state=42 ensures the same split every run (reproducibility)
X = df[all_features]
y = df["FloodProbability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set : {X_train.shape[0]} rows, {X_train.shape[1]} features")
print(f"Test  set : {X_test.shape[0]}  rows")

# Baseline XGBoost — default parameters
# objective='reg:squarederror' → regression task, minimises MSE internally
# n_jobs=-1 → use all CPU cores
baseline = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1
)
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)

# Metrics
# RMSE: average prediction error in the same units as FloodProbability
# R²  : fraction of variance explained by the model (1.0 = perfect)
rmse_base = float(np.sqrt(mean_squared_error(y_test, y_pred_base)))
r2_base   = float(r2_score(y_test, y_pred_base))

print(f"\nBaseline XGBoost (default parameters):")
print(f"  RMSE = {rmse_base:.5f}")
print(f"  R²   = {r2_base:.5f}")
print(f"\n  This is our reference — Step 4 (Optuna) should improve RMSE.")

# Save split and baseline scores
with open("data/split.pkl", "wb") as f:
    pickle.dump({
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "all_features": all_features
    }, f)

with open("data/baseline_scores.pkl", "wb") as f:
    pickle.dump({"rmse_base": rmse_base, "r2_base": r2_base}, f)

print("\nSplit saved to          : data/split.pkl")
print("Baseline scores saved to: data/baseline_scores.pkl")
print("=" * 60)
print("STEP 3 COMPLETE")
print("=" * 60)