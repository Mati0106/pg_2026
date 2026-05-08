# Flood Probability Prediction — ML Project

## Overview

Regression model predicting **FloodProbability** (continuous value 0–1) from 20 environmental and infrastructural features.

**Model:** XGBoost Regressor  
**Tuning:** Optuna (Bayesian Optimisation, 50 trials, 5-fold CV)  
**Explainability:** SHAP values

---

## Dataset note

`FloodProbability = mean(all feature columns) / 10` — **synthetic** dataset.  
`TotalRiskScore` (sum of all features) is **excluded** from feature engineering to avoid **data leakage**.

---

## Project structure

```
flood_project/
│
├── flood_data.csv
│
├── step1_load_data.py
├── step2_eda_features.py
├── step3_baseline_model.py
├── step4_optuna.py
├── step5_results.py
├── step6_feature_importance.py
├── step7_shap.py
│
├── data/
│   ├── flood_data_raw.pkl
│   ├── flood_data_featured.pkl
│   ├── split.pkl
│   ├── baseline_scores.pkl
│   ├── best_params.pkl
│   └── final_model.pkl
│
└── plots/
    ├── step5_results.png
    ├── step6_feature_importance.png
    ├── step7_shap_summary.png
    ├── step7_shap_waterfall.png
    └── step7_shap_dependence.png
```

---

## How to run

Run scripts **in order**. Each step saves output to `data/` and loads input from there.

```bash
python step1_load_data.py
python step2_eda_features.py
python step3_baseline_model.py
python step4_optuna.py
python step5_results.py
python step6_feature_importance.py
python step7_shap.py
```

---

## Requirements

```
pandas numpy matplotlib scikit-learn xgboost optuna shap
```

```bash
pip install pandas numpy matplotlib scikit-learn xgboost optuna shap
```

---

## What each step does

| Step | File | Description |
|------|------|-------------|
| 1 | `step1_load_data.py` | Load CSV. Print shape, dtypes, missing values, target stats. Save `flood_data_raw.pkl`. |
| 2 | `step2_eda_features.py` | Variance, Pearson correlation, IQR outlier detection. Create 4 interaction terms and 3 thematic group means. Save `flood_data_featured.pkl`. |
| 3 | `step3_baseline_model.py` | 80/20 train-test split. XGBoost with default parameters. Save baseline RMSE and R² to `baseline_scores.pkl`. |
| 4 | `step4_optuna.py` | Optimise 7 XGBoost hyperparameters over 50 Bayesian trials, each scored with 5-fold CV. Save `best_params.pkl`. |
| 5 | `step5_results.py` | Train final model with best params. Compute RMSE, MAE, R² on test set. Plot Predicted vs Actual and Residuals. Save `final_model.pkl`. |
| 6 | `step6_feature_importance.py` | MDI (Mean Decrease in Impurity) — built-in XGBoost feature importance. Top 15 plot. |
| 7 | `step7_shap.py` | SHAP values: global summary plot, single-row waterfall, dependence plot for top feature. |

---

## Key concepts

**Regression vs Classification**  
`FloodProbability` is continuous → regression problem. Classification predicts a category (flood / no flood).

**Data leakage**  
`TotalRiskScore = sum(features)` is proportional to `FloodProbability` — excluded so the model cannot recover the target formula directly.

**Why XGBoost?**  
Gradient boosting builds trees sequentially, each correcting errors of the previous one. Works well on tabular data, handles non-linear relationships, requires no feature scaling.

**Why Optuna over GridSearch?**  
GridSearch tests every parameter combination — exponentially slow. Optuna uses Bayesian Optimisation: each trial informs the next, so 50 intelligent trials outperform thousands of random ones.

**Why SHAP over MDI?**  
MDI reports global feature importance only. SHAP adds direction of effect, magnitude per observation, and is robust to correlated features. Based on Shapley values from game theory.

**Why no standardisation?**  
XGBoost splits on thresholds, not distances — feature scale is irrelevant. Standardisation is required for distance-based algorithms (KNN, SVM, linear regression).
