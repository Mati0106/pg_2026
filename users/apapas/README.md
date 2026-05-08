# Predykcja Market Value - Fortune 500

Projekt ML - predykcja wartosci rynkowej firm z listy Fortune 500 na podstawie danych finansowych.

Dane: https://www.kaggle.com/datasets/mirzayasirabdullah07/fortune-500-companies-us/data

## Struktura

```
Main.py                     - glowny skrypt (uruchamia caly pipeline)
data/fortune500.csv         - dane zrodlowe
src/scripts/eda.py          - analiza danych + wizualizacje
src/preprocessing/           - czyszczenie danych, feature engineering
src/modelling/models.py     - trenowanie modeli (LR, Lasso, RF, XGBoost)
src/modelling/optimization.py - optymalizacja hiperparametrow (Optuna TPE)
src/scripts/interpretacja.py  - SHAP, PDP, feature importance
wyniki/                     - wygenerowane wykresy
```

## Co robi projekt

1. **EDA** - histogramy, scatter ploty, boxploty
2. **Feature engineering** - Profit_Margin, Revenue_Per_Employee, Asset_Turnover, transformacje log
3. **4 modele** - Linear Regression (baseline), Lasso, Random Forest, XGBoost
4. **Optymalizacja** - Optuna (TPE, 50 trials, cv=5) dla RF, XGBoost i Lasso
5. **Interpretacja** - feature importance, wartosci Shapleya (SHAP), Partial Dependence Plots

## Wymagania

- Python 3.10+
- Biblioteki: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, shap, optuna

## Uruchomienie

XGBoost wymaga biblioteki OpenMP. Na Macu:
```bash
brew install libomp
```

```bash
# utworz srodowisko wirtualne
python3.10 -m venv venv1

# aktywuj 
source venv1/bin/activate

# zainstaluj pakiety
pip install -r requirements.txt

# uruchom projekt
python Main.py
```

Wykresy trafiaja do folderu `wyniki/`.
