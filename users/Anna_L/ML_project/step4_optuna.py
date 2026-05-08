# STEP 4 — HYPERPARAMETER OPTIMISATION (Optuna)
# Optuna replaces GridSearch — it uses Bayesian Optimisation, which learns
# from each trial and focuses the search on promising parameter regions.
# Each trial is evaluated with 5-fold cross-validation for stability.
# Saves: data/best_params.pkl

import pickle
import warnings
warnings.filterwarnings("ignore")

import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Load split from step 3
with open("data/split.pkl", "rb") as f:
    split = pickle.load(f)

X_train = split["X_train"]
y_train = split["y_train"]

print("=" * 60)
print("STEP 4 — HYPERPARAMETER OPTIMISATION (Optuna, 50 trials, CV=5)")
print("=" * 60)

# Objective function
# Optuna calls this function repeatedly, each time proposing different values
# for the parameters within the specified ranges.
# We return the cross-validated RMSE — Optuna minimises it.
def objective(trial):
    params = {
        "objective":        "reg:squarederror",
        "random_state":     42,
        "n_jobs":           -1,
        # Parameters Optuna will search
        # n_estimators    : number of trees (more = more powerful but slower)
        # max_depth       : tree depth (deeper = more complex, risk of overfitting)
        # learning_rate   : step size (smaller = more careful, needs more trees)
        # subsample       : fraction of rows used per tree (adds randomness)
        # colsample_bytree: fraction of features used per tree (adds randomness)
        # min_child_weight: minimum samples in a leaf (prevents overfitting)
        # gamma           : minimum loss reduction to make a split (regularisation)
        "n_estimators":     trial.suggest_int("n_estimators",     100, 800),
        "max_depth":        trial.suggest_int("max_depth",          3,   8),
        "learning_rate":    trial.suggest_float("learning_rate",  0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample",       0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree",0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight",   1,  10),
        "gamma":            trial.suggest_float("gamma",            0.0, 5.0),
    }
    scores = cross_val_score(
        XGBRegressor(**params),
        X_train, y_train,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    return -scores.mean()

# Run optimisation
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\nBest parameters found by Optuna:")
for k, v in study.best_params.items():
    print(f"  {k}: {round(v, 5) if isinstance(v, float) else v}")
print(f"\nBest RMSE (cross-validation): {study.best_value:.5f}")

# Save best parameters
with open("data/best_params.pkl", "wb") as f:
    pickle.dump({"best_params": study.best_params, "best_cv_rmse": study.best_value}, f)

print("\nBest parameters saved to: data/best_params.pkl")
print("=" * 60)
print("STEP 4 COMPLETE")
print("=" * 60)