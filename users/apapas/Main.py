"""
Predykcja Market Value firm Fortune 500
Uruchomienie: python Main.py
"""

import warnings
warnings.filterwarnings("ignore")

from src.scripts.eda import run_eda
from src.preprocessing.preprocessing import run_preprocessing
from src.modelling.models import run_modelling
from src.modelling.optimization import run_optimization, run_gridsearch
from src.scripts.interpretacja import run_interpretation


print("=" * 50)
print("  PREDYKCJA MARKET VALUE - FORTUNE 500")
print("=" * 50)

# 1. EDA
print("\n\n>>> ETAP 1: ANALIZA DANYCH")
df = run_eda()

# 2. Preprocessing
print("\n\n>>> ETAP 2: PREPROCESSING")
X_train, X_test, y_train, y_test, scaler, features = run_preprocessing(df)

# 3. Modele
print("\n\n>>> ETAP 3: MODELOWANIE")
results, models, predictions = run_modelling(X_train, X_test, y_train, y_test)

# 4. Optymalizacja
print("\n\n>>> ETAP 4: OPTYMALIZACJA (Optuna)")
best_model, best_name, results_opt, *_ = run_optimization(
    X_train, X_test, y_train, y_test, results, n_trials=30
)

# 4b. GridSearchCV
print("\n\n>>> ETAP 4b: OPTYMALIZACJA (GridSearchCV)")
results_gs = run_gridsearch(X_train, X_test, y_train, y_test, results_opt)

# 5. Interpretacja
print("\n\n>>> ETAP 5: INTERPRETACJA")
run_interpretation(best_model, X_test, features)

print("\n" + "=" * 50)
print(f"Najlepszy model: {best_name}")
print(f"Wykresy w folderze: wyniki/")
print("=" * 50)
