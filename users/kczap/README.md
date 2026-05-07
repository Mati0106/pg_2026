## **Honey Purity Prediction**

### **Cel projektu**

Ocenia czystość miodu na podstawie jego właściwości.
  
### Właściwości miodu

* CS (Color Score) — skala 1–10 (im wyżej tym ciemniejszy miód)
* Density — gęstość (g/cm³)
* WC (Water Content) — zawartość wody
* pH — kwasowość
* EC (Electrical Conductivity) — przewodność (mS/cm)
* F (Fructose) — poziom fruktozy
* G (Glucose) — poziom glukozy
* Pollen_analysis — pyłek 
* Viscosity — lepkość
* Purity — wartość docelowa (0.01–1.00)
* Price — cena wyliczona (jest w danych, ale z niej nie korzytamy)

### Przykład danych

| CS   | Density | WC    | pH   | Pollen_analysis | Viscosity | Purity | Price  |
|------|---------|-------|------|-----------------|-----------|--------|--------|
| 2.81 | 1.75    | 23.04 | 6.29 | Blueberry       | 4844.50   | 0.68   | 645.24 |
| 9.47 | 1.82    | 17.50 | 7.20 | Alfalfa         | 6689.02   | 0.89   | 385.85 |
| 4.61 | 1.84    | 23.72 | 7.31 | Chestnut        | 6883.60   | 0.66   | 639.64 |
| 1.77 | 1.40    | 16.61 | 4.01 | Blueberry       | 7167.56   | 1.00   | 946.46 |
| 6.11 | 1.25    | 19.63 | 4.82 | Alfalfa         | 5125.44   | 1.00   | 432.62 |


### Projekt obejmuje 
  * **XGBoost Regressor** do predykcji ciągłej wartości Purity
  * **XGBoost Classifier** do klasyfikacji Purity na klasy 0/1/2
  * **SHAP** do interpretacji modelu
  * **Optuna** do optymalizacji hiperparametrów
  * **LogisticRegression** do porówniania z XGBoost


### XGBoost Regressor

1. MSE: 0.0004838397193937689
2. Final Model MSE: 0.00047379835210978067 (Optuna)

* Model regresyjny osiąga bardzo wysoką dokładność  (MSE ≈ 0.00048)


### XGBoost Classifier

Accuracy: 98.15%

              precision    recall  f1-score   support

           0       0.98      0.99      0.99     27779
           1       0.99      0.97      0.98     23288
           2       0.98      0.99      0.98     30741

    accuracy                           0.98     81808
    macro avg       0.98      0.98      0.98     81808
    weighted avg       0.98      0.98      0.98     81808



### LogisticRegression

Accuracy: 46.91%

              precision    recall  f1-score   support

           0       0.52      0.64      0.57     27779
           1       0.43      0.29      0.35     23288
           2       0.43      0.45      0.44     30741

    accuracy                           0.47     81808
    macro avg       0.46      0.46      0.45     81808
    weighted avg       0.46      0.47      0.46     81808

* Model klasyfikacyjny działa, ale nie oddaje złożoności danych tak dobrze jak regresja.
* Klasa 0 jest przewidywana najlepiej (f1 = 0.57), co sugeruje, że model częściej trafia w próbki o najniższej czystości.
* Klasa 1 wypada najsłabiej (f1 = 0.35), co wskazuje na trudność w odróżnieniu próbek średniej czystości od pozostałych.
* Klasa 2 jest umiarkowanie rozpoznawana (f1 = 0.44).

### SHAP Summary Plot

Najważniejsze cechy wpływające na predykcję.

* CS
* Density
* WC
* pH
* G
* F

### SHAP Waterfall Plot

Wartości cech wpływające na predykcję.

* CS: −0.09
* pH: −0.05
* Density: −0.03
* WC: −0.02
* F: +0.01
* G: +0.01

**wysoka fruktoza i glukoza → zwiększają przewidywaną czystość**

**wysokie pH, duża zawartość wody, niska gęstość → obniżają czystość**