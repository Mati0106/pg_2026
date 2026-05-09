from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PowerTransformer

DEFAULT_DATA_FILE = Path(__file__).resolve().parent / "dataset" / "predictive_maintenance.csv"


def find_dataset_path(path=None):
    if path is not None:
        candidate = Path(path)
        if candidate.exists():
            return candidate

    if DEFAULT_DATA_FILE.exists():
        return DEFAULT_DATA_FILE

    for candidate in Path(__file__).resolve().parent.rglob("predictive_maintenance.csv"):
        return candidate

    raise FileNotFoundError(
        f"Could not locate predictive_maintenance.csv in {Path(__file__).resolve().parent}"
    )


def load_data(path=None):
    data_path = find_dataset_path(path)
    return pd.read_csv(data_path)


def clean_data(df):
    df = df.copy()
    df.drop(columns=["Product ID"], inplace=True, errors="ignore")

    if "Failure Type" in df.columns:
        df.drop(columns=["Failure Type"], inplace=True)

    return df


def remove_outliers(df):
    df = df.copy()
    if "Rotational speed [rpm]" in df.columns:
        q1 = df["Rotational speed [rpm]"].quantile(0.25)
        q3 = df["Rotational speed [rpm]"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df["Rotational speed [rpm]"] >= lower) & (df["Rotational speed [rpm]"] <= upper)]

    if "Torque [Nm]" in df.columns:
        df = df[df["Torque [Nm]"] < 70]

    return df


def encode_features(df):
    df = df.copy()
    if "Type" in df.columns:
        encoder = LabelEncoder()
        df["Type"] = encoder.fit_transform(df["Type"])
    return df

def featur_eng(df):
    df = df(["Air temperature [K]"]-273)
    # Odjęcie temp
    # Log z rotation speed  - znormalizować
    # Transformacjaj kwatylowa dla tool wear
    print(df)


def prepare_dataset(df):
    df = clean_data(df)
    df = remove_outliers(df)
    df = encode_features(df)

    if "Target" not in df.columns:
        raise ValueError("Input data must contain a Target column.")

    X = df.drop(columns=["Target"])
    y = df["Target"].astype(int)
    return X, y


def split_and_scale(X, y, test_size=0.2, random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    transformer = PowerTransformer(method="yeo-johnson")
    x_train = transformer.fit_transform(x_train)
    x_test = transformer.transform(x_test)

    return x_train, x_test, y_train, y_test
