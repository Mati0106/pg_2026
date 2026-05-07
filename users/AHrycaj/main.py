# Wczytanie danych
# Analiza danych + preprocessing (usunięcie pustych wartości i duplikatów, sprawdzenie czy wszystkie dane w kolumnie są tego samego typu)
# Feature engineering
# Modelowanie (model regresji logistycznej. 0 (woda niezdatna się do picia) lub 1 (woda zdatna do picia))
# Optymalizacja
# Interpretacja wyników

from utils import *

def run():
    df_raw = read_and_describe_data()
    df_clean = analyse_and_clean_data(df_raw)
    X_train, X_test, y_train, y_test, feature_names = scale_and_split(df_clean)
    model = model_with_linear_regression(X_train, X_test, y_train, y_test, feature_names)
    verify_model_accuracy(model, X_test, feature_names)


if __name__ == "__main__":
    run()