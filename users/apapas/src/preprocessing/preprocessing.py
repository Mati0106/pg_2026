import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run_preprocessing(df):
    df = df.copy()
    print(f"Dane: {df.shape[0]} wierszy, {df.shape[1]} kolumn")

    # feature engineering - nowe cechy
    df["Profit_Margin"] = df["Profits"] / df["Revenues"]
    df["Revenue_Per_Employee"] = df["Revenues"] / df["Number of Employees"]
    df["Asset_Turnover"] = df["Revenues"] / df["Assets"]
    df["Rank_Change"] = df["Previous Rank"] - df["Rank"]
    df["Log_Revenues"] = np.log1p(df["Revenues"])
    df["Log_Assets"] = np.log1p(df["Assets"])

    print("Nowe cechy: Profit_Margin, Revenue_Per_Employee, Asset_Turnover, "
          "Rank_Change, Log_Revenues, Log_Assets")

    # usuwanie brakow i inf
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Market Value", "Revenues", "Profits", "Assets"])
    print(f"Po usunieciu brakow: {len(df)} wierszy")

    # przygotowanie X i y
    features = ["Revenues", "Profits", "Assets", "Number of Employees",
                "Revenue Change", "Profit_Margin", "Revenue_Per_Employee",
                "Asset_Turnover", "Log_Revenues", "Log_Assets"]

    df_model = df[features + ["Market Value"]].dropna()
    X = df_model[features]
    y = df_model["Market Value"]

    print(f"Cechy: {features}")
    print(f"Rozmiar: {X.shape}")

    # split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # skalowanie - fit tylko na train
    scaler = StandardScaler()

    X_train_array = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_array, columns=features, index=X_train.index)

    X_test_array = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_array, columns=features, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features
