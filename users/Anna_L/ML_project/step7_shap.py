# STEP 7 — SHAP VALUES
# SHAP = SHapley Additive exPlanations (from game theory).
# SHAP answers three questions MDI cannot:
#   1. Which features matter most?           → summary plot
#   2. In which direction does each feature push the prediction?
#   3. Why did the model predict THIS value for THIS specific row?  → waterfall
#
# Three plots:
#   summary_plot    — global importance + direction of effect for all features
#   waterfall_plot  — step-by-step explanation of one single prediction
#   dependence_plot — how the most important feature's SHAP value changes
#                     across its value range
# Saves: plots/step7_shap_summary.png
#        plots/step7_shap_waterfall.png
#        plots/step7_shap_dependence.png

import numpy as np
import pandas as pd
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

os.makedirs("plots", exist_ok=True)

# Load final model and test data           ← było: # ── Load final model and test data ───...
with open("data/final_model.pkl", "rb") as f:
    payload = pickle.load(f)

with open("data/split.pkl", "rb") as f:
    split = pickle.load(f)

model        = payload["model"]
all_features = payload["all_features"]
X_test       = split["X_test"]

print("=" * 60)
print("STEP 7 — SHAP VALUES")
print("=" * 60)

# Compute SHAP values                      ← było: # ── Compute SHAP values ───...
# TreeExplainer is optimised for tree-based models (XGBoost, RandomForest).
# We use a sample of 2000 rows — representative and faster than the full set.
X_sample    = X_test.sample(2000, random_state=42)
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

print(f"\nSHAP values computed for {len(X_sample)} rows.")

# Plot 1: Summary plot                     ← było: # ── Plot 1: Summary plot ───...
# Each dot = one observation.
# X-axis: SHAP value (positive = pushes prediction up, negative = pushes it down).
# Color : feature value (red = high, blue = low).
# Features sorted by mean absolute SHAP value (most important at top).
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False, plot_size=(10, 8))
plt.tight_layout()
plt.savefig("plots/step7_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSHAP summary plot saved to: plots/step7_shap_summary.png")

# Plot 2: Waterfall plot (one observation) ← było: # ── Plot 2: Waterfall plot ───...
# Starts at the model's expected value (average prediction across all training rows).
# Each bar shows how much one feature pushes the prediction up (+) or down (−).
# Ends at the final predicted value for this specific row.
shap_exp = shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_sample.iloc[0],
    feature_names=all_features
)
plt.figure()
shap.plots.waterfall(shap_exp, show=False)
plt.tight_layout()
plt.savefig("plots/step7_shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()
print("SHAP waterfall plot saved to: plots/step7_shap_waterfall.png")

# Plot 3: Dependence plot                  ← było: # ── Plot 3: Dependence plot ───...
# X-axis: raw value of the feature.
# Y-axis: SHAP value (its contribution to the prediction for that observation).
# Shows whether the relationship is linear, stepped, or non-linear.
mean_shap   = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=all_features
).sort_values(ascending=False)

top_feature = mean_shap.index[0]

plt.figure(figsize=(7, 5))
shap.dependence_plot(top_feature, shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig("plots/step7_shap_dependence.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"SHAP dependence plot ({top_feature}) saved to: plots/step7_shap_dependence.png")

# Summary table                            ← było: # ── Summary table ───...
print(f"\nTop 10 features by mean absolute SHAP value:")
print(mean_shap.head(10).round(5).to_string())

print("=" * 60)
print("STEP 7 COMPLETE")
print("=" * 60)
print("\nAll steps finished. Output files:")
print("  data/  — intermediate pickled objects")
print("  plots/ — all visualisations (steps 5, 6, 7)")