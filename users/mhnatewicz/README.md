# Klasyfikator płci na podstawie cech fizycznych

Ten projekt implementuje kompletny pipeline uczenia maszynowego do przewidywania płci na podstawie wzrostu i wagi.

## Charakterystyka danych wejściowych

Przed przystąpieniem do modelowania, dane wejściowe zostały poddane szczegółowej analizie:

* **Weryfikacja jakości:** zbiór danych został sprawdzony pod kątem brakujących wartości, wartości nieokreślonych oraz
  wartości odstających.
* **Jednostki miar:**
    * Wzrost (Height): **centymetry [cm]**
    * Waga (Weight): **kilogramy [kg]**

## Cel i przeznaczenie programu

Głównym celem programu jest stworzenie precyzyjnego modelu klasyfikacji binarnej (Kobieta/Mężczyzna). Program ma za
zadanie nie tylko dokonać predykcji, ale również zilustrować, w jaki sposób poszczególne cechy wpływają na wynik
końcowy, co jest kluczowe w analizach statystycznych i medycznych.

## Architektura i działanie programu

Program składa się z pięciu kluczowych etapów:

### 1. Pobieranie i wstępna obróbka danych

Dane pobrane z www.kaggle.com, link bezpośredni do strony:
https://www.kaggle.com/datasets/riteshswami08/height-and-weight-dataset-with-bmi-age-and-gender

System wczytuje dane z formatu `.csv`. Na tym etapie zapewniona jest obsługa błędów (np. brak pliku), co gwarantuje
wykonanie kodu.

### 2. Inżynieria cech (Feature Engineering) i standaryzacja

* **BMI (Body Mass Index):** Program oblicza wskaźnik BMI według wzoru:
  $$BMI = \frac{masa\_kg}{(wzrost\_m)^2}$$
  Dodanie tej cechy pozwala modelowi lepiej zrozumieć relację między wzrostem a wagą.
* **Podział danych:** zbiór jest dzielony w proporcji 8:2 na część treningową i testową.
* **Skalowanie:** zastosowano `StandardScaler`, aby ujednolicić skale wszystkich cech, co ma znaczenie dla poprawnego
  działania regresji logistycznej.

### 3. Optymalizacja hiperparametrów (Optuna)

Zamiast ręcznego dobierania parametrów, program wykorzystuje bibliotekę **Optuna** do przeprowadzenia optymalizacji.

* Szukany jest optymalny parametr `C` (odwrotność siły regularyzacji) dla regresji logistycznej.
* Proces wykorzystuje 3-krotną walidację krzyżową (`Cross-Validation`), co minimalizuje ryzyko overfittingu (
  przetrenowania).

### 4. Modelowanie.

Po znalezieniu najlepszych parametrów, trenowany jest ostateczny model `LogisticRegression`. Program generuje:

* Wskaźnik **Accuracy** (dokładność ogólna).
* **Raport klasyfikacji**, zawierający metryki *precision*, *recall* oraz *f1-score*.

### 5. Interpretacja modelu (SHAP)

Program wykorzystuje wartości SHAP (SHapley Additive exPlanations):

* **Summary Plot:** Wykres pokazujący, które cechy (Waga, Wzrost czy BMI) mają najsilniejszy wpływ na klasyfikację płci.
* **Waterfall Plot:** Szczegółowa analiza pojedynczego przypadku (pacjenta), pokazująca "walkę" cech o przesunięcie
  prawdopodobieństwa w stronę danej klasy.

## Wymagane biblioteki

Do uruchomienia projektu niezbędne są następujące biblioteki:

* `pandas` – praca nad danymi.
* `matplotlib` – wizualizacja wyników.
* `scikit-learn` – algorytmy uczenia maszynowego i preprocessing.
* `optuna` – automatyczna optymalizacja hiperparametrów.
* `shap` – interpretacja graficzna modelu.
