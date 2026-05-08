# Libraries Used in This Project

## Data
| Library | Purpose |
|---|---|
| **pandas** | Core data manipulation — loading CSVs, DataFrames, encoding, filtering |
| **numpy** | Numerical operations, array handling (used internally by most ML libraries) |

## Machine Learning
| Library | Purpose |
|---|---|
| **scikit-learn** (`sklearn`) | ML algorithms (Logistic Regression, Decision Tree, Random Forest), scaling, train/test split, cross-validation, metrics |
| **xgboost** | XGBoost algorithm — gradient boosting, not part of sklearn but compatible with its interface |
| **optuna** | Hyperparameter tuning using Bayesian optimization (TPE sampler) — replaces exhaustive GridSearchCV with intelligent trial-based search |

## Explainability
| Library | Purpose |
|---|---|
| **shap** | SHAP (Shapley) values — explains model predictions globally (which features matter most) and locally (why the model predicted churn for a specific customer) |

## Visualization
| Library | Purpose |
|---|---|
| **matplotlib** | Base plotting library — figure creation, saving charts, layout |
| **seaborn** | High-level charts built on top of matplotlib — heatmaps, countplots, boxplots |

## Data Analysis & Reporting
| Library | Purpose |
|---|---|
| **ydata-profiling** | Auto-generates full HTML EDA report — distributions, correlations, missing values |

## Infrastructure
| Library | Purpose |
|---|---|
| **kaggle** | Kaggle API client — downloads dataset programmatically |
| **python-dotenv** | Loads `.env` file (Kaggle credentials) into environment variables |

