import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
import optuna

PROCESSED_PATH = "src/scripts/data/processed.csv"
seed      = 42
test_size = 0.33
target    = "ev_adoption_rate"


def objective(trial, X, y):
    n_estimators     = trial.suggest_int("n_estimators", 50, 400)
    max_depth        = trial.suggest_int("max_depth", 1, 10)
    learning_rate    = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
    subsample        = trial.suggest_float("subsample", 0.6, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)

    model = XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=subsample,
        colsample_bytree=colsample_bytree, random_state=seed, verbosity=0
    )
    return cross_val_score(model, X, y, cv=3,
                           scoring="neg_root_mean_squared_error", n_jobs=-1).mean()


def run():
    df = pd.read_csv(PROCESSED_PATH)

    drop_cols = [c for c in [target, "ev_sales", "ev_stock"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    print("Train:", X_train.shape, "  Test:", X_test.shape)
    print("Features:", list(X.columns))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train),
                   n_trials=100, show_progress_bar=True)

    trial = study.best_trial
    print("Accuracy: {}".format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

    model = XGBRegressor(**trial.params, random_state=seed, verbosity=0)
    model.fit(X_train, y_train)

    joblib.dump(model, "src/modelling/xgb_model.pkl")
    joblib.dump((X_test, y_test), "src/modelling/test_data.pkl")
