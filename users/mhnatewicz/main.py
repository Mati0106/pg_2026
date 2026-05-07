# Importujemy wszystkie potrzebne funkcje z naszego pliku utils.py
from utils import (
    load_data,
    perform_feature_engineering,
    optimize_hyperparameters,
    train_and_evaluate_model,
    interpret_with_shap
)

# Wywołanie głównej logiki programu
if __name__ == "__main__":
    file_path = 'weight-height.csv'

    # 1. Pobranie danych
    df = load_data(file_path)

    if df is not None:
        # 2. Przygotowanie danych
        (X_tr_scaled, X_ts_scaled,
         X_tr_oryg, X_ts_oryg,
         y_tr, y_ts, scaler) = perform_feature_engineering(df)

        # 3. Optymalizacja hiperparametrów (Optuna)
        best_hyperparams = optimize_hyperparameters(X_tr_scaled, y_tr, n_trials=15)

        # 4. Trenowanie i ocena modelu
        final_model, test_predictions = train_and_evaluate_model(
            X_tr_scaled, y_tr, X_ts_scaled, y_ts, best_hyperparams
        )

        # 5. Wygenerowanie wykresów SHAP
        interpret_with_shap(final_model, X_tr_scaled, X_ts_scaled, X_ts_oryg, y_ts)