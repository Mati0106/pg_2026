
Temat projektu:
Określenie wagi człowieka na podstawie wzrostu oraz płci przy użyciu modeli uczenia maszynowego (Regresja).

Opis kroków zadania:

1. Wczytanie danych. 
   - Pobranie danych pliku weight-height.xls)
   - Przeprowadzono wstępną weryfikację struktury danych - wyświetlenie pierwszych wierszy zbioru

2. Diagnostyka danych i Inżynieria Cech
   - Zamieniamy płeć (Female/Male) na liczby (0/1), żeby model zrozumiał daną
   - Zastosowano funkcje .isnull().sum()do sprawdzenie czy są brakujące wartości w bazie danych oraz .describe() do weryfikacji statystyk min/max.
   - Wykonujemy wykres pudełkowy (boxplot), aby zweryfikować i przeanalizować wartości odstające. 
   - Wizualizacja danych odstających (boxplot). Wykres pudełkowy dla wzrostu
   - Wizualizacja danych odstających (boxplot). Wykres pudełkowy dla wag
   - Feature Engineering (FE): Przekształcenie danych kategorycznych (płeć) na numeryczne (0/1).
   - Podział danych: Zbiór podzielono na część treningową (80%) i testową (20%) przy użyciu train_test_split.
     zbiór treningowy to dane na których się model uczy a testowy to zbiór którego nie zna i na którym sprawdzę skuteczność uczenia.

3. Modelowanie danych (Modeling):
   - Wykorzystano model Lasu Losowego (RandomForestRegressor), który pozwala na uchwycenie nieliniowych zależności między wzrostem, płcią a wagą.
   - Modelowanie czyli uczenie na zbiorze treningowym przy parametrach domyślnych Lasu Losowego
   - Sprawdzenie jak sie model nauczył. Sprawdzamy błąd wartości. Jaka jest predykcja w stosunku do wartości rzeczywistych ze zbioru testowego.
   - wypisuję wartość błędu. R-squere współczynnik determinacji - w jakim stopniu model wyjaśnia zmienność danych.
     Wartość bliska 1 oznacza, że model niemal idealnie przewiduje rzeczywiste wartości ciężaru/wagi na podstawie wzrostu i płci
   - Zastosowano metryki R^2 oraz MSE na zbiorze testowym w celu oceny precyzji przewidywań.

4. Optymalizacja parametrów (Hyperparameter tuning) 
   - Nadal korzystamy z RandomForestRegressor ale zoptymalizujemy parametry. Poprzednie były domyślne.
   - Zastosuję GridSearchCV z walidacją krzyżową (CV), co pozwoli na automatyczne dobranie parametrów modelu (n_estimators, max_depth) w celu maksymalizacji jego skuteczności.
   - Sprawdzanie. Sprawdzamy błąd po zmianie/optymalizacji parametrów uczenia.
   -Zastosowano metryki R^2 oraz MSE (mean Squer Error) na zbiorze testowym w celu oceny precyzji przewidywań.

5. Wyniki (Results interpretation):
   - Interpretacja Feature Importance (FI): Określamy wpływ poszczególnych cech na decyzje modelu.
   - Interpretacja SHAP: Wizualizacja wpływu zmiennych na przewidywania (pokazuje wpływ wzrostu i płci na wagę) 
   - Ważność cech (Feature Importance): przedstawiam jaka została przyjęta waga dla wzrostu i płci przy określeniu ciężaru

