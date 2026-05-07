from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import shap

TARGET_FEATURE_NAME = 'Potability'

def read_and_describe_data():
    dataset_path = Path.cwd().joinpath('water_potability').joinpath('water_potability.csv')
    print(dataset_path)
    main_df = pd.read_csv(dataset_path.resolve())
    df = main_df.copy()
    print(df.describe())

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

    return df

def analyse_and_clean_data(df):
    df = df.copy()
    df_dropna = df.dropna(subset=df.columns, axis=0)
    df_drop_duplicates = df_dropna.drop_duplicates(subset=df.columns)
    print('Raw set: ', df.shape, 'Clean set:', df_drop_duplicates.shape)
    print(df.dtypes)
    return df_drop_duplicates

def analyse_data(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

def scale_and_split(df, target_name = ''):
    #Rozdzielamy zbiory na cechy i wartość celu (to, co będziemy potem przewidywać)
    X = df.drop(TARGET_FEATURE_NAME, axis=1)
    y = df[TARGET_FEATURE_NAME]

    #Wrzucamy nazwy kolumn, aby potem użyć ich na wykresach
    feature_names = X.columns

    #Skalowanie zmiennych jest krokiem wymaganym. Przesuwa ona średnią do wartosci 0 a odchylenie standardowe do przedziału <-1,1>
    scaler = StandardScaler()
    #Skalujemy cały zbiór obserwacji
    X = scaler.fit_transform(X)
    #Teraz gdy wszystkie cechy (kolumny) są już porównywalne - tj. mają wspólną średnią i odchylenie standardowe

    #Dzielimy zbiór na treningowy (X_train, y_train) i testowy (X_test, y_test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, feature_names



def model_with_linear_regression(X_train, X_test, y_train, y_test, feature_names):
    model_lg = LogisticRegression(max_iter=120, random_state=0)
    model_lg.fit(X_train, y_train)
    predictions = model_lg.predict(X_test)

    print(classification_report(y_test, predictions))

    print('Accuracy score for LogisticRegression:', accuracy_score(y_test, predictions))
    cm = confusion_matrix(y_test, predictions)

    print('Confusion matrix for LogisticRegression', cm)
    sns.heatmap(cm / np.sum(cm), annot=True, fmt='0.2%', cmap='Reds')

    return model_lg


def verify_model_accuracy(model, X_test, feature_names):
    # compute the SHAP values for the linear model
    explainer = shap.Explainer(model.predict, X_test)
    explainer.feature_names = feature_names
    shap_values = explainer(X_test)
    shap.plots.beeswarm(shap_values)

