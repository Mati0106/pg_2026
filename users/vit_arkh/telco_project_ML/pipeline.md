# Telco Customer Churn — ML Pipeline

**Problem:** Binary classification — predict `Churn = 1` (customer leaves) or `Churn = 0` (customer stays), based on 7043 customer records from a telecom company.

**Primary metric: Recall** — a missed churner (FN) costs full LTV=\$1000, a false alarm (FP) costs only \$100 discount. Catching churners is 10× more valuable than avoiding false alarms.

---

## Step 1 — Data Loading
- Download dataset from Kaggle API using credentials from `.env`
- Load into pandas DataFrame

## Step 2 — Initial Analysis (EDA)
- `ydata_profiling` generates a full HTML report saved to `reports/`
- Report includes distributions, correlations (Pearson/Spearman/Kendall), missing values
- Runs on **raw data** — key finding: `TotalCharges` is stored as `object` instead of `float`

## Step 3 — Business Charts (EDA)
- 5 seaborn charts saved to `charts/eda/` showing churn dependency on:
  - `Contract` — month-to-month customers churn far more
  - `InternetService` — Fiber Optic has surprisingly high churn
  - `tenure` — new customers (low tenure) leave most often
  - `MonthlyCharges` — higher bills correlate with higher churn
  - `PaymentMethod` — Electronic Check customers churn significantly more

## Step 4 — Preprocessing
At this stage we **don't yet know which columns are useful** — so we encode all of them:

| Step | What | How |
|---|---|---|
| Fix `TotalCharges` | Convert to float, fill 11 NaN with 0 | `pd.to_numeric()`, `fillna(0)` |
| Encode target | `Churn`: Yes→1, No→0 | `.map()` |
| Drop `customerID` | Not useful for modeling | `drop()` |
| Encode `gender` | Male→1, Female→0 | `.map()` |
| Encode binary cols | Yes→1, No→0 for `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` | `.map()` |
| One-hot encode | `PaymentMethod`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `MultipleLines` | `pd.get_dummies(drop_first=True)` |

## Step 5 — Feature Relationships
All columns are now numeric — correlations work on the full dataset:
- Pearson / Spearman / Kendall heatmaps saved to `charts/correlation/`
- Mutual Information scores (`mutual_info_classif`) saved to `charts/mutual_information.png`
- MI is the primary signal — captures non-linear dependencies, works better than correlation for binary targets

## Step 6 — Train-Test Split
- 80/20 split, `stratify=y` — preserves the 27%/73% churn ratio in both sets
- Test set is never seen during training

## Step 7 — Scaling (Logistic Regression only)
- `StandardScaler` fitted **once** on `X_train`, applied to both sets `X_train` and `X_test`
- Applied **only to Logistic Regression** — gradient-based(learns weights by making small iterative steps toward minimum error), sensitive to feature scale
- Tree-based models (Decision Tree, Random Forest, XGBoost) receive **raw unscaled data** — splits on thresholds, scale is irrelevant

## Step 8 — Baseline Model Training
4 models trained and compared, sorted by Recall:
- Logistic Regression uses **ElasticNet regularization** (`l1_ratio=1.0, solver='saga'`) — with `l1_ratio=1.0` it acts as pure Lasso (L1), zeroing out weak feature weights
- Tree-based models (Decision Tree, Random Forest, XGBoost) receive raw unscaled data

| Model | Accuracy | Recall | Precision | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | **0.8062** | **0.5615** | **0.6583** | **0.6061** |
| Random Forest | 0.7871 | 0.5027 | 0.6225 | 0.5562 |
| XGBoost | 0.7722 | 0.4973 | 0.5831 | 0.5368 |
| Decision Tree | 0.7374 | 0.4893 | 0.5055 | 0.4973 |

**Logistic Regression wins baseline** — simplest model outperforms all ensemble methods.

## Step 9 — Confusion Matrix (Logistic Regression)

|  | Predicted: Stays | Predicted: Churns |
|---|---|---|
| **Actual: Stays** | TN = 926 ✅ | FP = 109 ❌ |
| **Actual: Churns** | FN = 164 ❌ | TP = 210 ✅ |

**164 missed churners (FN)** — the most costly mistake.

## Step 10 — Hyperparameter Tuning
- **Optuna** (`n_trials=50`, `scoring='recall'`, TPE Bayesian sampler) replaces GridSearchCV
- Optuna learns from previous trials which parameter regions are promising — faster and searches continuous ranges
- `class_weight='balanced'` for sklearn models, `scale_pos_weight≈3` for XGBoost — address 27%/73% class imbalance
- LR tunes both `C` and `l1_ratio` (0.0=L2/Ridge vs 1.0=L1/Lasso) — Optuna picks the better regularization automatically
- Results saved to `data/tuned_model_results.csv`
- Best XGBoost model object returned directly — no need to re-load from CSV

| Model | Accuracy | Recall | Precision | F1-Score | Best Params |
|---|---|---|---|---|---|
| **XGBoost** | 0.6061 | **0.9626** | 0.3996 | 0.5647 | learning_rate=0.0138, max_depth=4, scale_pos_weight=5 |
| Decision Tree | 0.6529 | 0.8797 | 0.4256 | 0.5737 | max_depth=3, min_samples_split=46 |
| Random Forest | 0.7026 | 0.8422 | 0.4667 | 0.6006 | n_estimators=64, max_depth=3 |
| Logistic Regression | 0.7388 | 0.7834 | 0.5052 | 0.6143 | C=2.74, l1_ratio=0.0 |

**XGBoost wins on Recall (0.96)** — trade-off: lower Precision (0.40), more false alarms.

## Step 11 — Confusion Matrix (Tuned XGBoost)

|  | Predicted: Stays | Predicted: Churns |
|---|---|---|
| **Actual: Stays** | TN = 494 ✅ | FP = 541 ❌ |
| **Actual: Churns** | FN = 14 ✅ | TP = 360 ✅ |

**FN dropped from 164 → 14** — XGBoost misses almost no real churners.

## Step 12 — Business Impact Simulation

```
Net profit = TP × (LTV - discount) - FP × discount - FN × LTV
           = TP × 900              - FP × 100       - FN × 1000
```

Three strategies compared — including the real business baseline (no ML):

| | Give everyone a discount | Logistic Regression | Tuned XGBoost |
|---|---|---|---|
| Retained customers (TP) | 374 → +$336,600 | 210 → +$189,000 | 360 → +$324,000 |
| False alarms (FP) | 1035 → -$103,500 | 109 → -$10,900 | 541 → -$54,100 |
| Missed churners (FN) | 0 | 164 → -$164,000 | 14 → -$14,000 |
| **Net profit** | **$233,100** | **$14,100** | **$255,900** |

**XGBoost vs give-everyone-a-discount:** +\$22,800 — contacts 494 fewer customers (1035→541 false alarms), far more efficient at scale.  
**XGBoost vs Logistic Regression:** +\$241,800 — Logistic Regression is worst of all three.

## Step 13 — SHAP Feature Importance

Using the tuned XGBoost model to explain **which features drive predictions** and why:

- `shap_summary.png` — global beeswarm: which features matter most across all test customers
- `shap_dependence_top1.png` / `shap_dependence_top2.png` — dependence plots for the 2 most important features (selected dynamically by mean SHAP value)
- `shap_force.png` — local Force Plot for the first churner in the test set: which features pushed toward churn prediction
- `shap_waterfall.png` — Waterfall breakdown for the same churner

All charts saved to `charts/shap/`.
