# 1. Ładowanie danych
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Pobranie danych ( plik nazywa się weight-height.csv)
df = pd.read_csv('weight-height.xls')
print("Pierwsze 5 wierszy danych:")
print(df.head())

# 2. FE (Inżynieria Cech)
# Zamieniamy płeć (Female/Male) na liczby (0/1), żeby model to zrozumiał
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
X = df[['Height', 'Gender']]
y = df['Weight']

# Sprawdzenie brakujących wartości
print("Braki w danych:")
print(df.isnull().sum())

# Szybkie sprawdzenie statystyk (szukamy anomalii w min/max)
print("\nStatystyki danych (sprawdź czy min/max są logiczne):")
print(df.describe())

# Wizualizacja danych odstających (boxplot) 
import matplotlib.pyplot as plt
df.boxplot(column=['Height', 'Weight'])
plt.title("Wykres pudełkowy - sprawdzanie wartości odstających")
plt.show()

# Podział na zbiór treningowy i testowy (to jest kluczowe dla sprawdzenia, czy model się nauczył)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Modelowanie
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Optymalizacja (Hyperparameter tuning)
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
grid = GridSearchCV(RandomForestRegressor(), param_grid, cv=3)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# 5. Wyniki (FI + SHAP)
import shap
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

print("Ważność cech (Feature Importance):")
print(pd.Series(best_model.feature_importances_, index=X.columns))

# Wizualizacja SHAP (pokazuje wpływ wzrostu i płci na wagę)
shap.summary_plot(shap_values, X_test)