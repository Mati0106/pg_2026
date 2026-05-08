import numpy as np
import matplotlib.pyplot as plt
import optuna

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import os

from src.modelling.models import evaluate_model

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _plot_history(study, ax, title):
    values = [t.value for t in study.trials if t.value is not None]
    best = [max(values[:i + 1]) for i in range(len(values))]
    ax.plot(values, "o", alpha=0.5, label="Trial")
    ax.plot(best, "-", label="Best")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective Value")
    ax.set_title(title)
    ax.legend()


def _plot_importances(study, ax, title):
    importances = optuna.importance.get_param_importances(study)
    params = list(importances.keys())
    vals = list(importances.values())
    ax.barh(params, vals)
    ax.set_xlabel("Importance")
    ax.set_title(title)


def run_optimization(X_train, X_test, y_train, y_test, results_before, n_trials=30):
    os.makedirs("wyniki", exist_ok=True)

    # Random Forest
    def objective_rf(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_categorical("max_depth", [5, 10, 15, 20, None]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }
        model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
        return cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()

    print(f"Optymalizacja Random Forest ({n_trials} trials)...")
    study_rf = optuna.create_study(direction="maximize")
    study_rf.optimize(objective_rf, n_trials=n_trials)
    print(f"Najlepsze parametry RF: {study_rf.best_params}")
    print(f"R2 (CV): {study_rf.best_value:.4f}")

    # XGBoost
    def objective_xgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model = XGBRegressor(random_state=42, verbosity=0, **params)
        return cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()

    print(f"\nOptymalizacja XGBoost ({n_trials} trials)...")
    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(objective_xgb, n_trials=n_trials)
    print(f"Najlepsze parametry XGB: {study_xgb.best_params}")
    print(f"R2 (CV): {study_xgb.best_value:.4f}")

    # Lasso
    def objective_lasso(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 100.0, log=True)
        model = Lasso(alpha=alpha, max_iter=5000)
        return cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()

    print(f"\nOptymalizacja Lasso ({n_trials} trials)...")
    study_lasso = optuna.create_study(direction="maximize")
    study_lasso.optimize(objective_lasso, n_trials=n_trials)
    print(f"Najlepsze parametry Lasso: {study_lasso.best_params}")
    print(f"R2 (CV): {study_lasso.best_value:.4f}")

    best_rf = RandomForestRegressor(random_state=42, n_jobs=-1, **study_rf.best_params)
    best_rf.fit(X_train, y_train)

    best_xgb = XGBRegressor(random_state=42, verbosity=0, **study_xgb.best_params)
    best_xgb.fit(X_train, y_train)

    best_lasso = Lasso(max_iter=5000, **study_lasso.best_params)
    best_lasso.fit(X_train, y_train)

    print("\nPo optymalizacji:")
    results_after = []
    result_rf_opt = evaluate_model("Random Forest (opt)", best_rf, X_test, y_test)
    results_after.append(result_rf_opt)
    result_xgb_opt = evaluate_model("XGBoost (opt)", best_xgb, X_test, y_test)
    results_after.append(result_xgb_opt)
    result_lasso_opt = evaluate_model("Lasso (opt)", best_lasso, X_test, y_test)
    results_after.append(result_lasso_opt)

    # wykres przed vs po
    opt_model_names = ["Random Forest", "XGBoost", "Lasso"]
    r2_before = [r["R2"] for r in results_before if r["model"] in opt_model_names]
    r2_after = [r["R2"] for r in results_after]

    x = np.arange(3)
    w = 0.3
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w/2, r2_before, w, label="Przed", color="#4C72B0")
    ax.bar(x + w/2, r2_after, w, label="Po", color="#55A868")

    for i, (v1, v2) in enumerate(zip(r2_before, r2_after)):
        ax.text(i - w/2, v1 + 0.02, f"{v1:.3f}", ha="center")
        ax.text(i + w/2, v2 + 0.02, f"{v2:.3f}", ha="center")

    ax.set_ylabel("R2")
    ax.set_title("R2 przed i po optym. (Optuna)")
    ax.set_xticks(x)
    ax.set_xticklabels(opt_model_names)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("wyniki/optymalizacja_porownanie.png", dpi=150)
    plt.close()

    # historia optymalizacji
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _plot_history(study_rf,    axes[0], "RF - historia optymalizacji")
    _plot_history(study_xgb,   axes[1], "XGB - historia optymalizacji")
    _plot_history(study_lasso, axes[2], "Lasso - historia optymalizacji")
    plt.tight_layout()
    plt.savefig("wyniki/optuna_historia.png", dpi=150)
    plt.close()

    # waznosc parametrow
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _plot_importances(study_rf,    axes[0], "RF - waznosc parametrow")
    _plot_importances(study_xgb,   axes[1], "XGB - waznosc parametrow")
    _plot_importances(study_lasso, axes[2], "Lasso - waznosc parametrow")
    plt.tight_layout()
    plt.savefig("wyniki/optuna_parametry.png", dpi=150)
    plt.close()

    best_result = max(results_after, key=lambda r: r["R2"])
    best_name_map = {
        "Random Forest (opt)": (best_rf,    "Random Forest (optym)"),
        "XGBoost (opt)":       (best_xgb,   "XGBoost (optym)"),
        "Lasso (opt)":         (best_lasso, "Lasso (optym)"),
    }
    best_model, best_name = best_name_map[best_result["model"]]

    print(f"\nNajlepszy model: {best_name}")
    return best_model, best_name, results_after, study_rf, study_xgb, study_lasso
