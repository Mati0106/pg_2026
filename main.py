import datetime

import kaggle
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder

dataset = "akshaydattatraykhare/car-details-dataset"


kaggle.api.dataset_download_files(dataset, path=".", unzip=True)

print(f"Dataset {dataset} downloaded and extracted successfully.")
df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")


df["car_age"] = datetime.date.today().year - df["year"]
df["car_name"] = df["name"].apply(lambda x: x.split()[0])


plt.figure(figsize=(10, 4))
sns.boxplot(x=df["selling_price"], color="#3498db")
plt.title("Rozkład cen przed usunięciem wartości odstających", fontsize=14)
plt.xlabel("Cena sprzedaży (miliony PLN)")
plt.gca().xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x / 1_000_000:.1f} mln")
)
plt.tight_layout()
plt.show()

upper_limit = df["selling_price"].quantile(0.95)
df["selling_price"] = df["selling_price"].clip(upper=upper_limit)

plt.figure(figsize=(10, 4))
sns.boxplot(x=df["selling_price"], color="#3498db")
plt.title("Rozkład cen przed usunięciem wartości odstających", fontsize=14)
plt.xlabel("Cena sprzedaży")
plt.gca().xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x / 1_000_000:.1f} mln")
)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
sns.histplot(df["selling_price"], color="#e67e22")
plt.title("Histogram rozkładu ceny sprzedaży", fontsize=14)
plt.xlabel("Cena sprzedaży")
plt.gca().xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x / 1_000_000:.1f} mln")
)
plt.tight_layout()
plt.show()


# 2. Feature Encoding
# #Przez to model moze pomyslec ze car_name zyskuje zaleznosc na podstawie marki
# auta ale robie tak bo byloby za duzo kolumn true/false
le = LabelEncoder()
df["car_name"] = le.fit_transform(df["car_name"])

owner_map = {  # Im mniej ownerow tym lepiej
    "Test Drive Car": 0,
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
}

df["owner"] = df["owner"].map(owner_map)


# Nie mozemy matematycznie zakladac ze np. diesel jest lepszy od benzyny dlatego one-hot encoding
df = pd.get_dummies(
    df, columns=["fuel", "seller_type", "transmission"], drop_first=True
)


X = df.drop(["selling_price", "name", "year"], axis=1)
y = df["selling_price"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def objective(trial):
    param = {
        "verbosity": 0,
        "objective": "reg:absoluteerror",  # Używamy MAE jako funkcji straty, ponieważ jest bardziej odporny na wartości odstające niż MSE, czyli pojedyncze bardzo drogie auta mają mniejszy wpływ na ostateczny model
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),  # Ilość drzew
        "max_depth": trial.suggest_int("max_depth", 2, 8),  # Głębokość drzewa
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.2, log=True
        ),  # Szybkość uczenia się im mniejsza tym dokładniejszy ale wolniejszy model
    }

    kf = KFold(
        n_splits=5, shuffle=True, random_state=42
    )  # 5-krotna walidacja krzyżowa, shuffle=True dla losowego mieszania danych, random_state dla powtarzalności wyników
    mae_scores = []

    for train_idx, val_idx in kf.split(X_train):
        xt, xv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        yt, yv = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = xgb.XGBRegressor(**param)
        model.fit(xt, yt, eval_set=[(xv, yv)], verbose=False)
        preds = model.predict(xv)
        mae_scores.append(mean_absolute_error(yv, preds))

    return np.mean(mae_scores)


study = optuna.create_study(
    direction="minimize"
)  # Ustawiamy kierunek optymalizacji na minimalizację, ponieważ chcemy zminimalizować MAE (Mean Absolute Error) - im mniejszy, tym lepszy model.
study.optimize(objective, n_trials=50)

print("Best Trial:")
print(f"  Value (MAE): {study.best_trial.value}")
print(f"  Params: {study.best_trial.params}")

best_model = xgb.XGBRegressor(**study.best_trial.params)
best_model.fit(X_train, y_train)

test_preds = best_model.predict(X_test)
final_mae = mean_absolute_error(y_test, test_preds)

print(f"\nFinal Test MAE: {final_mae}")

plt.figure(figsize=(10, 5))
sns.regplot(
    x=y_test,
    y=test_preds,
    scatter_kws={"alpha": 0.3, "color": "#2c3e50"},
    line_kws={"color": "red"},
)
plt.title("Zależność: Cena Rzeczywista vs Przewidziana", fontsize=14)
plt.xlabel("Cena Rzeczywista")
plt.ylabel("Cena Przewidziana")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
importance_scores = best_model.get_booster().get_score(importance_type="gain")
importance_df = pd.Series(importance_scores).sort_values(ascending=False).head(10)

sns.barplot(x=importance_df.values, y=importance_df.index, palette="magma")
plt.title("Co najbardziej determinuje cenę? (Top 10 cech - Gain)", fontsize=14)
plt.xlabel("Wpływ na zmianę ceny (Średni Zysk/Gain)")
plt.ylabel("Cecha samochodu")
plt.tight_layout()
plt.show()


print("\nObliczanie wartości SHAP")
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("Wpływ cech na cenę (SHAP Summary)", fontsize=14)
plt.tight_layout()
plt.show()

print("\n--- WNIOSKI Z MODELU ---")

importance = pd.Series(importance_scores).sort_values(ascending=False)

top_feature = importance.index[0]

print(f"Głównym czynnikiem wpływającym na cenę samochodu jest: {top_feature.upper()}")

print(f"Najważniejsze parametry techniczne to: {', '.join(importance.index[:5])}")
