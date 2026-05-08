# external
import numpy as np
import optuna
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

# --- CONSTANTS ---
RANDOM_STATE = 42
N_OPTUNA_TRIALS = 50
OPTUNA_METRIC = "auc"


def modeling(X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test) -> dict:
    """Train benchmark (LogisticRegression) and baseline XGBoost. Return models + metrics."""
    results = {}

    # --- Benchmark: Logistic Regression (requires scaled data) ---
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]

    results["logistic_regression"] = {
        "model": lr,
        "accuracy": accuracy_score(y_test, lr_pred),
        "auc": roc_auc_score(y_test, lr_proba),
        "confusion_matrix": confusion_matrix(y_test, lr_pred),
        "report": classification_report(y_test, lr_pred),
    }

    # --- XGBoost baseline (uses unscaled data — tree-based, scale-invariant) ---
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

    results["xgboost_baseline"] = {
        "model": xgb_model,
        "accuracy": accuracy_score(y_test, xgb_pred),
        "auc": roc_auc_score(y_test, xgb_proba),
        "confusion_matrix": confusion_matrix(y_test, xgb_pred),
        "report": classification_report(y_test, xgb_pred),
    }

    return results


def optimize_with_optuna(X_train, y_train, n_trials: int = N_OPTUNA_TRIALS) -> dict:
    """Run Optuna hyperparameter search for XGBoost. Maximizes AUC-ROC via 3-fold CV."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
        }
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(
            model, X_train, y_train, cv=3, scoring="roc_auc", n_jobs=-1
        )
        return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"Best AUC (CV): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    return study.best_params
