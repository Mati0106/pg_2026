# Jira Bug Fix Prediction

Predicting whether a Jira issue will be fixed or rejected, using public Atlassian issue tracker data.

## Target Variable (Y)

`was_fixed` — binary classification:
- `1` = issue resolved as "Fixed"
- `0` = issue closed as Duplicate, Cannot Reproduce, Spam, Invalid, etc.

## Dataset

Source: [Jira Dataset — Kaggle](https://www.kaggle.com/datasets/cesaranasco/jira-dataset)  
File: `data/GFG_FINAL.csv`  
Rows used: 16,562 (resolved issues only)

## Project Structure

```
src/preprocessing.py   — load_data(), feature_engineering(), split_and_scale()
src/modeling.py        — modeling(), optimize_with_optuna()
src/interpretation.py  — feature_importance_plot(), shap_analysis()
notebooks/eda.ipynb    — Exploratory Data Analysis with conclusions
main.py                — full pipeline entry point
```

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

## Models

| Model | Role |
|---|---|
| Logistic Regression | Benchmark |
| XGBoost | Main model |
| XGBoost + Optuna | Hyperparameter-optimized model |

## Features

`priority_encoded`, `issue_type_encoded`, `votes`, `description_length`,
`has_labels`, `reporter_issue_count`, `project_key_encoded`,
`time_to_first_comment_days`, `comment_count`, `has_attachment`
