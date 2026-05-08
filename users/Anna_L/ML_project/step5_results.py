# STEP 5 — FINAL MODEL & RESULTS INTERPRETATION
# Train XGBoost with the best parameters from Optuna.
# Evaluate on the test set and produce diagnostic plots:
#   - Predicted vs Actual  → points should lie on the diagonal
#   - Residuals plot       → errors should be random around zero
# Saves: data/final_model.pkl, plots/step5_results.png

import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

os.makedirs("plots", exist_ok=True)

# Load data from previous steps
with open("data/split.pkl", "rb") as f:
    split = pickle.load(f)

with open("data/best_params.pkl", "rb") as f:
    tuning = pickle.load(f)

with open("data/baseline_scores.pkl", "rb") as f:
    baseline = pickle.load(f)

X_train      = split["X_train"]
X_test       = split["X_test"]
y_train      = split["y_train"]
y_test       = split["y_test"]
all_features = split["all_features"]
best_params  = tuning["best_params"]
rmse_base    = baseline["rmse_base"]
r2_base      = baseline["r2_base"]

print("=" * 60)
print("STEP 5 — FINAL MODEL & RESULTS INTERPRETATION")
print("=" * 60)

# Train final model with Optuna's best parameters
best_model = XGBRegressor(
    **best_params,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Metrics
# RMSE: average error in the same units as FloodProbability
# MAE : average absolute error (less sensitive to large errors than RMSE)
# R²  : 1.0 = perfect model; 0.0 = model no better than predicting the mean
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae  = float(mean_absolute_error(y_test, y_pred))
r2   = float(r2_score(y_test, y_pred))

print(f"\nTest set results:")
print(f"  RMSE = {rmse:.5f}  (average prediction error in target units)")
print(f"  MAE  = {mae:.5f}  (average absolute error)")
print(f"  R²   = {r2:.5f}  (fraction of variance explained by the model)")

print(f"\nImprovement over baseline:")
print(f"  RMSE : {rmse_base:.5f} → {rmse:.5f}  (Δ = {rmse_base - rmse:+.5f})")
print(f"  R²   : {r2_base:.5f} → {r2:.5f}   (Δ = {r2 - r2_base:+.5f})")

# Plot 1: Predicted vs Actual
# Points lying on the red dashed line = perfect predictions.
# Scatter around the line shows the magnitude of errors.

# Plot 2: Residuals
# Residual = actual - predicted.
# A good model has residuals randomly scattered around zero (no pattern).
# A pattern would indicate the model is systematically wrong in certain regions.

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(y_test, y_pred, alpha=0.3, s=5, color="steelblue")
axes[0].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--", lw=1.5, label="Perfect prediction"
)
axes[0].set_xlabel("Actual values")
axes[0].set_ylabel("Predicted values")
axes[0].set_title("Predicted vs Actual")
axes[0].legend()

residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.3, s=5, color="steelblue")
axes[1].axhline(0, color="red", lw=1.5, linestyle="--")
axes[1].set_xlabel("Predicted values")
axes[1].set_ylabel("Residuals (actual − predicted)")
axes[1].set_title("Residuals — should be random around zero")

plt.tight_layout()
plt.savefig("plots/step5_results.png", dpi=150)
plt.close()

print("\nPlot saved to: plots/step5_results.png")

# Save final model and predictions
with open("data/final_model.pkl", "wb") as f:
    pickle.dump({
        "model":        best_model,
        "y_pred":       y_pred,
        "all_features": all_features,
        "rmse":         rmse,
        "mae":          mae,
        "r2":           r2,
    }, f)

print("Final model saved to: data/final_model.pkl")
print("=" * 60)
print("STEP 5 COMPLETE")
print("=" * 60)