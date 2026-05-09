import json
from pathlib import Path

import optuna
import pandas as pd
from catboost import CatBoostClassifier
from data_pipeline import load_data, prepare_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier




def evaluate_model(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def baseline_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            criterion="entropy",
            max_depth=20,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=70,
            learning_rate=0.1,
            random_state=42,
            verbose=0,
        ),
    }


def objective_rf(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 5, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "random_state": 42,
        "n_jobs": -1,
    }
    model = RandomForestClassifier(**params)
    score = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), scoring="roc_auc", n_jobs=1)
    return score.mean()


def objective_xgb(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0),
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }
    model = XGBClassifier(**params)
    score = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), scoring="roc_auc", n_jobs=1)
    return score.mean()


def objective_catboost(trial, X, y):
    params = {
        "iterations": trial.suggest_int("iterations", 50, 250),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "random_strength": trial.suggest_float("random_strength", 1.0, 20.0),
        "border_count": trial.suggest_int("border_count", 32, 128),
        "verbose": 0,
        "random_state": 42,
    }
    model = CatBoostClassifier(**params)
    score = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), scoring="roc_auc", n_jobs=1)
    return score.mean()


def tune_model(name, X, y, objective, n_trials=25):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    return study


def compare_models():
    df = load_data()
    X, y = prepare_dataset(df)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    baseline = baseline_models()
    baseline_results = {}
    print("\n=== Baseline model performance ===")
    for name, model in baseline.items():
        metrics = evaluate_model(model, x_train, x_test, y_train, y_test)
        baseline_results[name] = metrics
        print(f"{name}: {metrics}")

    tuned_results = {}
    best_params = {}
    studies = {}
    tuning_config = [
        ("RandomForest", objective_rf),
        ("XGBoost", objective_xgb),
        ("CatBoost", objective_catboost),
    ]

    print("\n=== Running Optuna hyperparameter tuning ===")
    for name, objective in tuning_config:
        study = tune_model(name, X, y, objective, n_trials=25)
        studies[name] = study
        print(f"{name} best ROC AUC: {study.best_value:.5f}")
        best_params[name] = study.best_params

    print("\n=== Evaluating tuned models on holdout set ===")
    for name, params in best_params.items():
        if name == "RandomForest":
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        elif name == "XGBoost":
            model = XGBClassifier(**params, eval_metric="logloss", random_state=42, n_jobs=-1)
        elif name == "CatBoost":
            model = CatBoostClassifier(**params, verbose=0, random_state=42)
        else:
            continue

        tuned_results[name] = evaluate_model(model, x_train, x_test, y_train, y_test)
        print(f"{name}: {tuned_results[name]}")

    summary = []
    for name in baseline_results:
        base = baseline_results[name]
        tuned = tuned_results.get(name, {})
        summary.append(
            {
                "model": name,
                "base_accuracy": base["accuracy"],
                "tuned_accuracy": tuned.get("accuracy"),
                "base_f1": base["f1"],
                "tuned_f1": tuned.get("f1"),
                "base_roc_auc": base["roc_auc"],
                "tuned_roc_auc": tuned.get("roc_auc"),
            }
        )

    summary_df = pd.DataFrame(summary).set_index("model")
    print("\n=== Comparison table ===")
    print(summary_df)

    report_path = Path(__file__).resolve().parent / "optuna_tuning_summary.json"
    with report_path.open("w", encoding="utf-8") as out_file:
        json.dump({"baseline": baseline_results, "tuned": tuned_results, "best_params": best_params}, out_file, indent=2)
    print(f"\nSaved detailed results to {report_path}")

    return baseline_results, tuned_results, best_params


if __name__ == "__main__":
    compare_models()
