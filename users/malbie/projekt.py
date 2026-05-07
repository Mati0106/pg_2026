

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Pobranie danych ( plik nazywa się weight-height.xls)
df = pd.read_csv('weight-height.xls')
print("Pierwsze 5 wierszy danych:")
print(df.head())

# 2. FE (Inżynieria Cech)
# Zamieniamy płeć (Female/Male) na liczby (0/1), żeby model zrozumiał daną
df['Gender'] = df['Gender'].map({'Female': 1, 'Male': 0})

# Sprawdzenie cz są brakujące wartości w bazie danych
print("Braki w danych:")
print(df.isnull().sum())

# Szybkie sprawdzenie statystyk (szukamy anomalii w min/max)
print("\nStatystyki danych (sprawdź czy min/max są logiczne):")
print(df.describe())

# Wizualizacja danych odstających (boxplot). Wykres pudełkowy dla wzrostu
df.boxplot(column=['Height'])
plt.title("Wykres pudełkowy - sprawdzanie wartości odstających we wzroście")
plt.show()

# Wizualizacja danych odstających (boxplot). Wykres pudełkowy dla wagi
df.boxplot(column=['Weight'])
plt.title("Wykres pudełkowy - sprawdzanie wartości odstających w wadze osób")
plt.show()

X = df[['Height', 'Gender']]
y = df['Weight']

# Podział na zbiór treningowy i testowy (treningowy to dane na których się uczy a testowy to zbiór którego nie zna)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Modelowanie czyli uczenie na zbiorze treningowym przy parametrach domyślnych Lasu losowego
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Sprawdzenie jak sie model nauczył. Sprawdzamy błąd wartości.
# Jaka jest predykcja w stosunku do wartości rzeczywistych ze zbioru testowego.
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#wypisuje wartość błędu. R-squere współczynnik determinacji w jakim stopniu model wyjaśnia zmienność danych.
# Wartość bliska 1 oznacza, że model niemal idealnie przewiduje rzeczywiste wartości na podstawie wzrostu i płci
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# 4. Optymalizacja (Hyperparameter tuning).
# Nadal korzystamy z RandomForest ale zoptymalizujemy parametry. Poprzednie były domyślne
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
grid = GridSearchCV(RandomForestRegressor(), param_grid, cv=3)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

#Sprawdzanie best_model. Sprawdzamy błąd po zmianie parametrów uczenia.
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Best Model Mean Squared Error: {mse}')
print(f'Best Model R-squared: {r2}')

# 5. Wykres Shapleya który przedstawia jaka została przyjeta waga dla wzrostu i płci przy okresleniu ciężaru
import shap
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

print("Ważność cech (Feature Importance):")
print(pd.Series(best_model.feature_importances_, index=X.columns))

# Wizualizacja SHAP (pokazuje wpływ wzrostu i płci na wagę)
shap.summary_plot(shap_values, X_test)