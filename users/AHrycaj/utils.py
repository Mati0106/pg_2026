from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import shap

import warnings
warnings.filterwarnings('ignore')

TARGET_FEATURE_NAME = 'Potability'

def read_and_describe_data():
    dataset_path = Path.cwd().joinpath('water_potability').joinpath('water_potability.csv')
    print(dataset_path)
    main_df = pd.read_csv(dataset_path.resolve())
    df = main_df.copy()
    print(df.describe())

    # Tworzymy większe okno wykresu (np. szerokość 12, wysokość 10)
    fig, ax = plt.subplots(figsize=(12, 10))

    # Rysujemy mapę ciepła z nieco mniejszą czcionką wartości (annot_kws)
    sns.heatmap(
        df.corr(),
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        annot_kws={"size": 9},  # mniejsza czcionka liczb w kwadracikach
        ax=ax
    )

    # --- TUTAJ JEST MAGIA NAPRAWIAJĄCA NAPISY ---

    # Obracamy podpisy na osi X o 45 stopni i wyrównujemy do prawej
    plt.xticks(rotation=45, ha='right', fontsize=10)

    # Podpisy na osi Y zostawiamy poziomo (0 stopni), ale też lekko zmniejszamy czcionkę
    plt.yticks(rotation=0, fontsize=10)

    plt.title("Macierz korelacji cech", fontsize=14, pad=20)

    # Ta funkcja automatycznie dba o to, żeby napisy nie uciekały poza krawędzie obrazka
    plt.tight_layout()

    plt.show()

    return df

def analyse_and_clean_data(df):
    df_dropna = df.copy().dropna(subset=df.columns, axis=0)
    df_drop_duplicates = df_dropna.drop_duplicates(subset=df.columns)
    print('Raw set: ', df.shape, 'Clean set:', df_drop_duplicates.shape)
    print(df_drop_duplicates.dtypes)
    return df_drop_duplicates

def scale_and_split(df, target_name = ''):
    #Rozdzielamy zbiory na cechy i wartość celu (to, co będziemy potem przewidywać)
    X = df.drop(TARGET_FEATURE_NAME, axis=1)
    y = df[TARGET_FEATURE_NAME]

    #Wrzucamy nazwy kolumn, aby potem użyć ich na wykresach
    feature_names = X.columns

    #Dzielimy zbiór na treningowy (X_train, y_train) i testowy (X_test, y_test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('X_train shape:', X_train.shape, 'y_train shape:', y_train.shape)
    print('X_test shape:', X_test.shape, 'y_test shape:', y_test.shape)

    #Skalowanie zmiennych jest krokiem wymaganym. Przesuwa ona średnią do wartosci 0 a odchylenie standardowe do przedziału <-1,1>
    scaler = StandardScaler()
    #Skalujemy cały zbiór obserwacji
    X_train = scaler.fit_transform(X_train)  # fit_transform tylko na train
    X_test = scaler.transform(X_test)  # tylko transform na test!

    return X_train, X_test, y_train, y_test, feature_names





def model_with_logistic_regression(X_train, X_test, y_train, y_test, feature_names):
    model_lg = LogisticRegression()
    model_lg.fit(X_train, y_train)
    predictions = model_lg.predict(X_test)

    print(classification_report(y_test, predictions))
    # Precyzja (Precision): Mierzy, jaki odsetek przewidzianych wyników pozytywnych był faktycznie pozytywny.
    # Odpowiada na pytanie: "Jak często model ma rację, gdy przewiduje daną klasę?".
    #
    # Czułość (Recall/Sensitivity): Wskazuje, jaki odsetek rzeczywistych wyników pozytywnych został poprawnie wykryty przez model.
    # Odpowiada na pytanie: "Jaki procent wszystkich pozytywnych przypadków odnalazł model?".
    #
    # F1-score: Jest to średnia harmoniczna precyzji i czułości.
    # Stanowi balans między tymi dwoma miarami i jest szczególnie użyteczny, gdy klasy są niezrównoważone.
    #
    # Wsparcie (Support): Określa liczbę rzeczywistych wystąpień każdej klasy w zbiorze testowym.
    #
    # Dokładność (Accuracy): Ogólny odsetek poprawnych przewidywań dla wszystkich klas.

    print('Accuracy score for LogisticRegression:', accuracy_score(y_test, predictions))
    # Dokładność (Accuracy): Ogólny odsetek poprawnych przewidywań dla wszystkich klas.
    # Ponawiam, bo to często wykorzystywany parametr przy porównywaniu modeli

    cm = confusion_matrix(y_test, predictions)
    print('Confusion matrix for LogisticRegression\n', cm)
    ##Confusion matrix ma postać
    # [
    #   [ TN, FP ]
    #   [ FN, TP ]
    # ]
    # True-Positive (TP – prawdziwie pozytywna): przewidywanie pozytywne, faktycznie zaobserwowana klasa pozytywna
    #   (np. pozytywny wynik testu do picia wody i woda zdatna do picia)
    # True-Negative (TN – prawdziwie negatywna): przewidywanie negatywne, faktycznie zaobserwowana klasa negatywna
    #   (np. negatywny wynik testu do picia wody i woda niezdatna do picia)
    # False-Positive (FP – fałszywie pozytywna): przewidywanie pozytywne, faktycznie zaobserwowana klasa negatywna
    #   (np. pozytywny wynik testu do picia wody, jednak faktycznie woda niezdatna do picia)
    # False-Negative (FN – fałszywie negatywna): przewidywanie negatywne, faktycznie zaobserwowana klasa pozytywna
    #   (np. negatywny wynik testu do picia wody, jednak woda zdatna do picia)
    return model_lg


def verify_model_accuracy(model, X_train, X_test, feature_names):
    # compute the SHAP values for the linear model
    explainer = shap.Explainer(model.predict, X_train)
    explainer.feature_names = feature_names

    shap_values = explainer(X_test)
    # print (shap_values.shape)

    shap.plots.beeswarm(shap_values)

    #shap.plots.waterfall(shap_values[0])

    shap.plots.bar(shap_values)


