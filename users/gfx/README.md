# Projekt ML - przewidywanie ocen filmów IMDb

Model przewiduje ocenę filmu na IMDb na podstawie cech takich jak liczba głosów, gatunek, długość, przychody i rok premiery.

## Zawartość projektu

- `IMDB_FINAL_do_wysylki.ipynb` - notebook z całą analizą
- `app.py` - aplikacja Streamlit do rekomendacji filmów
- `imdb_top_1000.csv` - dataset Top 1000 filmów z IMDb
- `movie_metadata.csv` - dataset IMDB 5000 z Kaggle
- `requirements.txt` - lista wymaganych bibliotek
- `app_files/` - folder z zapisanymi plikami modelu (tworzy się po uruchomieniu notebooka)

## Co zawiera notebook

1. Wczytanie danych
2. EDA (rozkład ocen, ocena vs liczba głosów)
3. Czyszczenie danych i uzupełnianie braków
4. Feature engineering: one-hot encoding gatunków, top reżyserzy, rok premiery
5. Regresja liniowa i Random Forest jako baseline
6. XGBoost
7. Cross-validation
8. Analiza błędów na zbiorze testowym
9. Hyperparameter tuning - GridSearchCV
10. Learning curve
11. Połączenie z drugim datasetem (IMDB 5000) i ponowny trening
12. OMDb API - pobieranie plakatów dla aplikacji
13. Optuna - tuning Bayesian
14. SHAP - wyjaśnienie predykcji
15. Kod aplikacji Streamlit

## Wymagania

Python 3.10+ oraz biblioteki wymienione w `requirements.txt`. Instalacja:

```
pip install -r requirements.txt
```

## Uruchomienie

### Notebook

```
jupyter notebook
```

Następnie otworzyć `IMDB_FINAL_do_wysylki.ipynb` i wybrać Cell -> Run All. Najwięcej czasu zajmuje sekcja Optuna (50 prób cross-validation) i pobieranie plakatów przez OMDb API.

Sekcja OMDb wymaga osobistego klucza API. Darmowy klucz można uzyskać tutaj: https://www.omdbapi.com/apikey.aspx (plan FREE, limit 1000 zapytań na dobę). Klucz wklejam w komórce zamiast `TWOJ_KLUCZ_TUTAJ`.

### Aplikacja Streamlit

Notebook musi zostać wcześniej uruchomiony do końca, żeby utworzyć pliki w folderze `app_files/`. Następnie:

```
streamlit run app.py
```

Aplikacja otwiera się w przeglądarce na localhost:8501. Sidebar z filtrami (gatunki, dekada, długość, minimalna liczba głosów, Metacritic). Po kliknięciu "Znajdź filmy" wytrenowany model przewiduje oceny i wyświetla wyniki z plakatami pobranymi przez OMDb.

## Wyniki modeli

| Model | RMSE | R² |
|---|---|---|
| Regresja liniowa | 0.218 | 0.260 |
| Random Forest v1 | 0.193 | 0.421 |
| Random Forest v2 (więcej cech) | 0.187 | 0.455 |
| XGBoost (Top 1000) | 0.195 | 0.406 |
| XGBoost (5493 filmy) | 0.781 | 0.536 |
| XGBoost po Optunie | 0.777 | - |

Po połączeniu datasetów RMSE wzrosło ponieważ skala ocen się rozszerzyła (z 7.6-9.3 do 1.6-9.3), ale R² się poprawiło ponieważ większy zbiór ma więcej zmienności do wyjaśnienia.

## Wnioski z analizy

- Liczba głosów na IMDb jest najsilniejszym predyktorem oceny - filmy popularne otrzymują wyższe oceny
- Stare filmy są wyżej oceniane niż nowsze (efekt klasyków, które przetrwały próbę czasu)
- Bycie filmem akcji obniża predykcję, dramaty ją podbijają
- Cecha "topowy reżyser" praktycznie nie wpływa na model, bo jest skorelowana z innymi cechami
- Wysokie przychody box office obniżają predykcję dla niektórych filmów (efekt blockbusterów ocenianych niżej przez krytyków)

## Struktura projektu

```
.
├── IMDB_FINAL_do_wysylki.ipynb
├── app.py
├── requirements.txt
├── imdb_top_1000.csv
├── movie_metadata.csv
├── app_files/                      # tworzy się po uruchomieniu notebooka
│   ├── model.pkl
│   ├── movies.pkl
│   ├── top_directors.pkl
│   ├── top_genres.pkl
│   ├── features.pkl
│   └── history.json                # tworzy się po pierwszym uruchomieniu aplikacji
└── README.md
```

## Źródła datasetów

- Top 1000 IMDb: https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows
- IMDB 5000: https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset
