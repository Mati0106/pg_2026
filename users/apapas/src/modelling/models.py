import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os


def evaluate_model(name, model, X_test, y_test):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print(f"{name:20s} MAE: {mae:>10,.0f}  RMSE: {rmse:>10,.0f}  R2: {r2:.4f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "pred": pred}


def plot_prediction_scatter(ax, y_test, pred, name, color):
    ax.scatter(y_test, pred, alpha=0.5, color=color, s=30)
    max_val = max(y_test.max(), max(pred))
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1)
    ax.set_xlabel("Wartosc rzeczywista (mln$)")
    ax.set_ylabel("Predykcja (mln$)")
    ax.set_title(name)


def run_modelling(X_train, X_test, y_train, y_test):
    os.makedirs("wyniki", exist_ok=True)

    # 1) Linear Regression - baseline
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    result_lr = evaluate_model("Lin. Regression", lr, X_test, y_test)

    # 2) Lasso
    lasso = Lasso(alpha=1.0, max_iter=5000)
    lasso.fit(X_train, y_train)
    result_lasso = evaluate_model("Lasso", lasso, X_test, y_test)

    # 3) Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    result_rf = evaluate_model("Random Forest", rf, X_test, y_test)

    # 4) XGBoost
    xgb = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                        random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    result_xgb = evaluate_model("XGBoost", xgb, X_test, y_test)

    # 5) Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    result_ridge = evaluate_model("Ridge", ridge, X_test, y_test)

    # 6) LightGBM
    lgbm = LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                         random_state=42, verbosity=-1)
    lgbm.fit(X_train, y_train)
    result_lgbm = evaluate_model("LightGBM", lgbm, X_test, y_test)

    results = [result_lr, result_lasso, result_rf, result_xgb, result_ridge, result_lgbm]
    predictions = {
        "Lin. Regression": result_lr["pred"],
        "Lasso":           result_lasso["pred"],
        "Random Forest":   result_rf["pred"],
        "XGBoost":         result_xgb["pred"],
        "Ridge":           result_ridge["pred"],
        "LightGBM":        result_lgbm["pred"],
    }

    df_results = pd.DataFrame(results)
    best = df_results.loc[df_results["R2"].idxmax()]
    print(f"\nNajlepszy: {best['model']} (R2={best['R2']:.4f})")

    # wykres porownawczy
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["blue", "orange", "red", "green", "purple", "brown"]

    axes[0].bar(df_results["model"], df_results["R2"], color=colors)
    axes[0].set_title("R2 Score")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(df_results["model"], df_results["MAE"], color=colors)
    axes[1].set_title("MAE (mln $)")
    axes[1].tick_params(axis="x", rotation=15)

    axes[2].bar(df_results["model"], df_results["RMSE"], color=colors)
    axes[2].set_title("RMSE (mln $)")
    axes[2].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig("wyniki/porownanie_modeli.png", dpi=150)
    plt.close()

    # predykcje vs rzeczywistosc (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(21, 10))
    plot_items = [
        ("Lin. Regression", result_lr["pred"],    "blue"),
        ("Lasso",           result_lasso["pred"],  "orange"),
        ("Random Forest",   result_rf["pred"],     "red"),
        ("XGBoost",         result_xgb["pred"],    "green"),
        ("Ridge",           result_ridge["pred"],  "purple"),
        ("LightGBM",        result_lgbm["pred"],   "brown"),
    ]
    for ax, (name, pred, color) in zip(axes.flat, plot_items):
        plot_prediction_scatter(ax, y_test, pred, name, color)

    plt.suptitle("Predykcje v. rzeczywistosc", fontsize=14)
    plt.tight_layout()
    plt.savefig("wyniki/predykcje_vs_rzeczywiste.png", dpi=150)
    plt.close()

    for name, pred in predictions.items():
        max_err = np.max(np.abs(y_test.values - pred))
        print(f"  {name}: max blad = {max_err:,.0f} mln $")

    models = {
        "Lin. Regression": lr,
        "Lasso":           lasso,
        "Random Forest":   rf,
        "XGBoost":         xgb,
        "Ridge":           ridge,
        "LightGBM":        lgbm,
    }
    return results, models, predictions
