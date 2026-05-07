import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import optuna
import shap
import warnings

# Wyłączenie ostrzeżeń dla większej czytelności w konsoli.
warnings.filterwarnings('ignore')


def load_data(file_path: str) -> pd.DataFrame:
    print("1. Pobieranie danych...")
    try:
        df = pd.read_csv(file_path)
        print(f"Pomyślnie wczytano {len(df)} wierszy.")
        return df
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku {file_path}!")
        return None


def perform_feature_engineering(df: pd.DataFrame):
    print("\n2. Przeprowadzanie Feature Engineering...")

    # Feature Engineering: Tworzymy nową zmienną BMI
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)

    # Wybór cech (X) i celu (y)
    X = df[['Height', 'Weight', 'BMI']]
    y = df['Gender']

    # Dzielimy zbiór na zbiór treningowy (80%) i testowy (20%).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standaryzacja
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(
        f"Dodano cechę 'BMI'. Dane podzielono na treningowe ({len(X_train_scaled)}) i testowe ({len(X_test_scaled)}).")

    return X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, scaler


def optimize_hyperparameters(X_train_scaled, y_train, n_trials=20) -> dict:
    print("\n3. Rozpoczęcie optymalizacji hiperparametrów (Optuna)...")

    def objective(trial):
        C_param = trial.suggest_float('C', 1e-3, 10.0, log=True)
        model = LogisticRegression(C=C_param, max_iter=1000, random_state=42)
        score = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='accuracy').mean()
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    print(f"Optymalizacja zakończona! Najlepsze parametry: {best_params}")
    return best_params


def train_and_evaluate_model(X_train_scaled, y_train, X_test_scaled, y_test, best_params):
    print("\n4. Trenowanie i ocena ostatecznego modelu...")

    model = LogisticRegression(**best_params, max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\nOstateczna Dokładność (Accuracy): {accuracy:.4f}")

    print("\n--- Raport Klasyfikacji ---")
    print(classification_report(y_test, predictions))

    return model, predictions


def interpret_with_shap(model, X_train_scaled, X_test_scaled, X_test_oryginal, y_test):
    print("\n5. Generowanie interpretacji graficznej SHAP...")

    explainer = shap.Explainer(model, X_train_scaled, feature_names=X_test_oryginal.columns)
    shap_values = explainer(X_test_scaled)

    shap_values.data = X_test_oryginal.values

    print("-> Wyświetlanie wykresu podsumowującego (zamknij okno wykresu, by kontynuować)...")
    plt.figure()
    plt.title("Wpływ poszczególnych cech na decyzje modelu")
    shap.summary_plot(shap_values, features=X_test_oryginal, show=False)
    plt.tight_layout()
    plt.show()

    pacjent_id = 0
    rzeczywista = y_test.iloc[pacjent_id]

    print(f"\n-> Wyświetlanie decyzji dla pacjenta nr {pacjent_id} (Rzeczywista płeć: {rzeczywista})")
    plt.figure()
    shap.plots.waterfall(shap_values[pacjent_id], show=False)
    plt.tight_layout()
    plt.show()
    print("\nKoniec procesu! Model jest gotowy do wdrożenia.")