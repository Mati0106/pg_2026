Predykcja Cen Samochodów Używanych (Regresja)
Temat projektu:
Określenie rynkowej wartości samochodu na podstawie jego parametrów technicznych przy użyciu modeli uczenia maszynowego.

Opis kroków zadania:

Wczytanie danych.

Pobranie danych z pliku data.csv.

Przeprowadzono wstępną weryfikację struktury danych – wyświetlenie pierwszych wierszy zbioru.

Diagnostyka danych i Inżynieria Cech

Czyszczenie danych: usunięcie jednostek tekstowych (km, cm3) i rzutowanie na wartości liczbowe.

Zastosowano funkcję .isnull().sum() do sprawdzenia brakujących wartości oraz .describe() do weryfikacji statystyk.

Eksploracyjna Analiza Danych (EDA): Wykonano zaawansowany raport korelacji przy użyciu ydata-profiling.

Porównanie korelacji Pearsona i Spearmana w celu wykazania nieliniowości trendów cenowych.

Feature Engineering (FE): Przekształcenie danych kategorycznych (marka, model, paliwo) na numeryczne przy użyciu One-Hot Encoding.

Podział danych: Zbiór podzielono na część treningową (80%) i testową (20%) przy użyciu train_test_split.

Modelowanie danych (Modeling):

Wykorzystano model XGBoost Regressor, który pozwala na uchwycenie skomplikowanych, nieliniowych zależności rynkowych.

Modelowanie, czyli uczenie na zbiorze treningowym przy parametrach bazowych.

Sprawdzenie skuteczności nauki: porównanie predykcji z wartościami rzeczywistymi ze zbioru testowego.

Zastosowano metryki MAE (średni błąd) oraz R-square (współczynnik determinacji) do oceny precyzji.

Optymalizacja parametrów (Hyperparameter tuning):

Zastosowano framework Optuna do automatycznego i inteligentnego doboru parametrów modelu.

Przeprowadzono 20 prób optymalizacji (trials) w celu minimalizacji błędu predykcji.

Weryfikacja błędu po optymalizacji parametrów uczenia.

Wyniki (Results interpretation):

Interpretacja Feature Importance: Określenie, które cechy techniczne najczęściej wpływały na podziały w drzewach decyzyjnych.

Interpretacja SHAP: Wizualizacja wpływu zmiennych na konkretne wyceny (wyjaśnienie kierunku wpływu wieku, przebiegu i marki na cenę).

Wdrożenie: Przygotowanie interaktywnej aplikacji w systemie Streamlit do szybkiej wyceny pojazdów przez użytkownika.