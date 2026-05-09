# Global Electric Vehicle Dataset 2023

**Dataset:** [Kaggle — Global Electric Vehicle Dataset 2023](https://www.kaggle.com/datasets/padmapiyush/global-electric-vehicle-dataset-2023)

**Cel:** Analiza i prognozowanie adopcji pojazdow w pelni elektrycznych (BEV) na podstawie danych swiatowych, z dedykowana prognoza dla Polski do 2030 roku.

## Podejscie

Model trenowany jest na danych globalnych (wszystkie kraje, tylko BEV), dzieki czemu uczy sie ogolnych wzorcow adopcji — jak rok, tempo wzrostu i specyfika kraju wplywaja na wskaznik adopcji BEV. Dane globalne zapewniaja wystarczajaca liczbe obserwacji do wiarygodnego treningu modelu (XGBoost).

Po wytrenowaniu model jest uzywany do prognozy dla Polski — kraju o niskiej adopcji BEV. Profil Polski (zakodowany kraj, historyczne tempo wzrostu) jest przekazywany do modelu, ktory na tej podstawie przewiduje trajektorie adopcji do 2030 roku. Takie podejscie pozwala odpowiedziec na pytanie: jesli Polska bedzie rozwijac sie zgodnie ze swiatowymi wzorcami, gdzie znajdzie sie w 2030?

## Kroki projektu

1. Wczytanie danych
2. Analiza danych (EDA)
3. Preprocessing i feature engineering
4. Model (XGBoost)
5. Optymalizacja hiperparametrow (Optuna + GridSearchCV)
6. Interpretacja wynikow (Feature Importance, SHAP, PDP)
7. Prognoza dla Polski do 2030

## Setup

### 1. Klonuj repozytorium i przejdz do folderu projektu

```bash
git clone <repo-url>
cd project
```

### 2. Stworz i aktywuj wirtualne srodowisko

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Zainstaluj zaleznosci

```bash
pip install -r requirements.txt
```

### 4. Dodaj klucz API Kaggle

Stworz plik `.env` w folderze projektu (na podstawie `.env.example`):

```
KAGGLE_API_TOKEN=twoj_token_tutaj
```

Token mozna wygenerowac na: https://www.kaggle.com/settings → API → Create New Token

### 5. Uruchom

**Notebook (zalecane):**
```bash
jupyter notebook analysis.ipynb
```

**Pipeline (terminal):**
```bash
python Main.py
```

## Struktura projektu

```
project/
├── src/
│   ├── preprocessing/   # wczytanie, filtrowanie, feature engineering
│   ├── modelling/       # trening modeli, optymalizacja
│   └── scripts/         # EDA, ewaluacja, interpretacja
├── analysis.ipynb       # glowny notebook
├── Main.py              # pipeline terminalowy
├── requirements.txt
├── .env.example
└── README.md
```
