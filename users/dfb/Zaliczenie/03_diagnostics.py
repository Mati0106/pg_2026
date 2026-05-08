import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
import numpy as np

# 1. WCZYTANIE DANYCH I MODELU
df_ml = pd.read_csv('data/cleaned_data.csv')
X = df_ml.drop('price_in_pln', axis=1)
y = df_ml['price_in_pln']

# Wczytujemy model zapisany przez Optunę
model = xgb.XGBRegressor()
model.load_model('data/car_model.json')

print("Generowanie wykresów diagnostycznych...")

# --- WYKRES 1: WAŻNOŚĆ CECH (XGBoost) ---
plt.figure(figsize=(10, 8))
xgb.plot_importance(model, ax=plt.gca(), max_num_features=15, importance_type='weight')
plt.title("Co najbardziej wpływa na cenę? (Ważność cech)")
plt.tight_layout()

# --- WYKRES 2: RZECZYWISTOŚĆ VS PREDYKCJA ---
plt.figure(figsize=(8, 6))
sample_idx = np.random.choice(len(y), 500, replace=False)
y_actual = y.iloc[sample_idx]
y_pred = model.predict(X.iloc[sample_idx])

plt.scatter(y_actual, y_pred, alpha=0.5, color='royalblue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Cena rzeczywista (PLN)")
plt.ylabel("Cena przewidziana przez AI (PLN)")
plt.title("Jak blisko prawdy jest model?")
plt.grid(True, linestyle='--', alpha=0.6)

# --- WYKRES 3: ANALIZA SHAP  ---
print("Liczenie SHAP... To może potrwać ok. 15-30 sekund. Czekaj cierpliwie!")
explainer = shap.Explainer(model)
# Obliczamy wartości SHAP dla małej próbki, żeby nie trwało to wieków
X_sample = X.sample(100, random_state=42)
shap_values = explainer(X_sample)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, show=False)
plt.title("Analiza SHAP - wpływ parametrów na konkretną wycenę")

plt.show()