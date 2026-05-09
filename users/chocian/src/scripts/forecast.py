import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.preprocessing.preprocess import (
    COL_REGION, COL_YEAR, COL_CHARGING,
    PROCESSED_PATH, REGION_MAP_PATH,
)

FORECAST_YEARS = list(range(2024, 2031))
POLAND = "Poland"


def _project_series(series, years, fallback_growth=0.05):
    """Project a numeric series forward using average growth from the last 3 periods."""
    last_val = series.iloc[-1]
    deltas = series.pct_change().dropna()
    deltas = deltas[deltas != 0].tail(3)
    avg_growth = deltas.mean() if not deltas.empty else fallback_growth

    projected = []
    val = last_val
    for _ in years:
        val = val * (1 + avg_growth)
        projected.append(val)
    return projected


def run():
    model = joblib.load("src/modelling/xgb_model.pkl")
    df = pd.read_csv(PROCESSED_PATH)
    region_map = pd.read_csv(REGION_MAP_PATH)
    region_to_code = dict(zip(region_map["region"], region_map["code"]))

    poland_code = region_to_code.get(POLAND, -1)
    if poland_code == -1:
        print("Brak danych dla Polski w zbiorze!")
        return

    poland_rows = df[df[COL_REGION] == poland_code].sort_values(COL_YEAR)
    if poland_rows.empty:
        print("Brak wierszy dla Polski po preprocessingu!")
        return

    last_row = poland_rows.iloc[-1].copy()
    avg_yoy = poland_rows["yoy_growth"].tail(3).mean()
    projected_cs = _project_series(poland_rows[COL_CHARGING], FORECAST_YEARS)

    # build future feature rows aligned to model's expected columns
    feature_cols = model.feature_names_in_
    future_rows = []
    for year, cs in zip(FORECAST_YEARS, projected_cs):
        row = last_row[feature_cols].copy()
        row[COL_YEAR] = year
        row["yoy_growth"] = avg_yoy
        row[COL_CHARGING] = cs
        future_rows.append(row)

    X_future = pd.DataFrame(future_rows)[feature_cols]
    predictions = model.predict(X_future)

    historical = poland_rows.groupby(COL_YEAR)["ev_adoption_rate"].mean()

    # --- plot 1: BEV adoption rate forecast ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        historical.index, historical.values,
        marker="o", color="steelblue", label="Dane historyczne (Polska)",
    )
    ax.plot(
        FORECAST_YEARS, predictions,
        marker="o", linestyle="--", color="orange", label="Prognoza XGBoost",
    )
    ax.axvline(x=2023, color="gray", linestyle=":", label="Granica prognozy")
    ax.set_title("Prognoza adopcji BEV w Polsce do 2030")
    ax.set_xlabel("Rok")
    ax.set_ylabel("Adopcja BEV (%)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend()
    plt.tight_layout()
    plt.show()

    # --- plot 2: projected charging stations ---
    hist_cs = poland_rows.groupby(COL_YEAR)[COL_CHARGING].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        hist_cs.index, hist_cs.values,
        marker="o", color="darkorange", label="Dane historyczne",
    )
    ax.plot(
        FORECAST_YEARS, projected_cs,
        marker="o", linestyle="--", color="red", label="Projekcja",
    )
    ax.axvline(x=2023, color="gray", linestyle=":")
    ax.set_title("Projekcja liczby stacji ladowania EV w Polsce do 2030")
    ax.set_xlabel("Rok")
    ax.set_ylabel("Liczba stacji ladowania")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend()
    plt.tight_layout()
    plt.show()

    # --- numeric summary ---
    print("\nPrognoza adopcji BEV w Polsce (XGBoost):")
    for year, val in zip(FORECAST_YEARS, predictions):
        print(f"  {year}: {val:.4f}%")

    print("\nProjekcja stacji ladowania (Polska):")
    for year, val in zip(FORECAST_YEARS, projected_cs):
        print(f"  {year}: {val:,.0f}")
