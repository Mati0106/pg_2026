import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. Wczytujemy dane i tworzymy nowy plik
df_ml = pd.read_csv('data/cleaned_data.csv')

# 2. Podział na X i y
X = df_ml.drop('price_in_pln', axis=1)
y = df_ml['price_in_pln']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


3# Przygotowanie danych (zmiana na cyfry)
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'random_state': 42
    }

    model = xgb.XGBRegressor(**param)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)


# 4. Uruchomienie poszukiwań
print("Optuna szuka najlepszych parametrów... Może to chwilę potrwać.")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print(f"Najlepszy błąd (MAE): {study.best_value}")
print(f"Najlepsze parametry: {study.best_params}")

# 5. Trenujemy finalny model na najlepszych ustawieniach
best_model = xgb.XGBRegressor(**study.best_params)
best_model.fit(X_train, y_train)

# 6. ZAPIS -  dla aplikacji Streamlit!
best_model.save_model('data/car_model.json')
print("Model zapisany do data/car_model.json")