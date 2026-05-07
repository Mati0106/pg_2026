import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import optuna
from users.kczap.profiling_data.load_honey_dataset import load_honey_data

df = load_honey_data()
X = df.drop(["Purity", "Pollen_analysis"], axis=1)
y = df["Purity"]

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)


#Optuna objective function
def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10)
    }

    # Train XGBoost model with suggested parameters
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return mse

# Run optimization for 50 trials
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

#best hyperparameters and best MSE found
print("Best params:", study.best_params)
print("Best MSE:", study.best_value)
