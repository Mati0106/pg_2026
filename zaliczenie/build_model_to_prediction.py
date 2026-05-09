import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             precision_recall_curve, recall_score, roc_curve)
from xgboost import XGBClassifier

from data_pipeline import load_data, prepare_dataset, split_and_scale, featur_eng
from eda import visualize_data


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def build_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            criterion="entropy",
            max_depth=20,
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.1, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=70, learning_rate=0.1, random_state=42, verbose=0
        ),
    }

    try:
        import lightgbm as lgb
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(
            n_estimators=50, learning_rate=0.1, random_state=42
        )
    except ImportError:
        logger.info("LightGBM is not installed; skipping LightGBM model.")

    return models


def evaluate_model(model, x_train, x_test, y_train, y_test, plot=False):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": auc(*roc_curve(y_test, y_prob)[:2]),
    }

    if plot:
        _plot_diagnostics(model, x_test, y_test, y_pred, y_prob)

    return metrics


def _plot_diagnostics(model, x_test, y_test, y_pred, y_prob):
    class_names = ["not fail", "fail"]
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix: {model.__class__.__name__}")
    plt.show()

    print(classification_report(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker="o", color="blue")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.show()


def run_model_pipeline(data_path=None, plot=False):
    df = load_data(data_path)
    X, y = prepare_dataset(df)
    x_train, x_test, y_train, y_test = split_and_scale(X, y)

    models = build_models()
    results = []

    for model_name, model in models.items():
        logger.info("Training %s", model_name)
        metrics = evaluate_model(model, x_train, x_test, y_train, y_test, plot=plot)
        results.append({"model": model_name, **metrics})

    results_df = pd.DataFrame(results).set_index("model")
    logger.info("\n%s", results_df)

    return results_df


def main():
    script_path = Path(__file__).resolve().parent
    candidate_path = script_path / "dataset" / "predictive_maintenance.csv"
    data_path = candidate_path if candidate_path.exists() else None
    df = load_data(data_path)
    featur_eng(df)
    visualize_data(df)
    run_model_pipeline(data_path)


if __name__ == "__main__":
    main()
