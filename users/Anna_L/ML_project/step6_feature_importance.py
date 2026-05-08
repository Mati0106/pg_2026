# STEP 6 — FEATURE IMPORTANCE (MDI)
# MDI = Mean Decrease in Impurity (also called Gini Importance).
# For every split in every tree, XGBoost records which feature was used
# and how much it reduced impurity. MDI sums these reductions per feature.
#
# Limitation: MDI tends to favour features with many unique values and
# can be biased when features are correlated → SHAP (Step 7) is more reliable.
# Saves: plots/step6_feature_importance.png

import pandas as pd
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

# Load final model from step 5
with open("data/final_model.pkl", "rb") as f:
    payload = pickle.load(f)

with open("data/split.pkl", "rb") as f:
    split = pickle.load(f)

model        = payload["model"]
all_features = payload["all_features"]

print("=" * 60)
print("STEP 6 — FEATURE IMPORTANCE (MDI)")
print("=" * 60)

# Compute importance
# feature_importances_ returns an array of MDI scores, one per feature.
# Values sum to 1.0 — a score of 0.25 means the feature accounts for 25%
# of total impurity reduction across all trees.
importance = pd.Series(
    model.feature_importances_,
    index=all_features
).sort_values(ascending=False)

print("\nTop 15 features (MDI):")
print(importance.head(15).round(4).to_string())

print(f"\nFeatures with zero importance (unused by model):")
zero = importance[importance == 0]
print(zero.index.tolist() if len(zero) > 0 else "  None")

# Plot
plt.figure(figsize=(9, 7))
importance.head(15).sort_values().plot(
    kind="barh", color="steelblue", edgecolor="white"
)
plt.xlabel("Importance (MDI)")
plt.title("Feature Importance — XGBoost (top 15)")
plt.tight_layout()
plt.savefig("plots/step6_feature_importance.png", dpi=150)
plt.close()

print("\nPlot saved to: plots/step6_feature_importance.png")
print("\nNOTE: MDI can be biased toward correlated or high-cardinality features.")
print("      Use SHAP values (Step 7) for a more reliable interpretation.")
print("=" * 60)
print("STEP 6 COMPLETE")
print("=" * 60)