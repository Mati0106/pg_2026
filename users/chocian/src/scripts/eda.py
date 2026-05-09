import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing.preprocess import (
    COL_REGION, COL_YEAR, COL_VALUE, COL_PARAMETER,
    PARAM_SALES, PARAM_CHARGING,
)


def run(df_bev, df_raw=None):
    print(df_bev.shape)
    print(df_bev.dtypes)
    print("Braki danych:\n", df_bev.isnull().sum())
    print(df_bev.describe().round(2))

    df_sales = df_bev[df_bev[COL_PARAMETER] == PARAM_SALES]

    # 1. Global BEV sales trend over the years
    trend = df_sales.groupby(COL_YEAR)[COL_VALUE].sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(trend.index, trend.values, marker="o", color="steelblue")
    ax.set_title("Laczna sprzedaz BEV na swiecie w kolejnych latach")
    ax.set_xlabel("Rok")
    ax.set_ylabel("Liczba sprzedanych BEV")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()

    # 2. Year-over-year growth (global)
    yoy = trend.pct_change() * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(yoy.index, yoy.values, color="steelblue")
    ax.set_title("Rok do roku - wzrost sprzedazy BEV na swiecie (%)")
    ax.set_xlabel("Rok")
    ax.set_ylabel("Wzrost (%)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()

    # 3. Top 10 countries by total BEV sales
    top10 = (
        df_sales.groupby(COL_REGION)[COL_VALUE]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    plt.figure(figsize=(10, 5))
    top10.plot(kind="bar", color="steelblue")
    plt.title("Top 10 krajow wedlug sprzedazy BEV")
    plt.ylabel("Liczba sprzedanych BEV")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # 4. Poland vs top 5 countries
    pivot = df_sales.pivot_table(
        index=COL_YEAR, columns=COL_REGION, values=COL_VALUE, aggfunc="sum"
    )
    highlight = top10.head(5).index.tolist()
    if "Poland" not in highlight:
        highlight.append("Poland")

    fig, ax = plt.subplots(figsize=(12, 6))
    for country in highlight:
        if country in pivot.columns:
            style = {"linewidth": 2.5, "linestyle": "--"} if country == "Poland" else {}
            ax.plot(pivot.index, pivot[country], marker="o", label=country, **style)
    ax.set_title("BEV — Polska vs liderzy swiatowi (sprzedaz)")
    ax.set_xlabel("Rok")
    ax.set_ylabel("Liczba sprzedanych BEV")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend()
    plt.tight_layout()
    plt.show()

    # 5. Charging stations trend (uses unfiltered raw data)
    if df_raw is not None:
        df_cs = df_raw[df_raw[COL_PARAMETER] == PARAM_CHARGING]
        if not df_cs.empty:
            cs_trend = df_cs.groupby(COL_YEAR)[COL_VALUE].sum()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(cs_trend.index, cs_trend.values, marker="o", color="darkorange")
            ax.set_title("Laczna liczba publicznych stacji ladowania EV na swiecie")
            ax.set_xlabel("Rok")
            ax.set_ylabel("Liczba stacji ladowania")
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            plt.tight_layout()
            plt.show()

            # Top 10 countries by charging stations
            top10_cs = (
                df_cs.groupby(COL_REGION)[COL_VALUE]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )
            plt.figure(figsize=(10, 5))
            top10_cs.plot(kind="bar", color="darkorange")
            plt.title("Top 10 krajow wedlug liczby stacji ladowania EV")
            plt.ylabel("Liczba stacji ladowania")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()

            # Charging stations vs BEV sales correlation (per region-year)
            bev_agg = (
                df_sales.groupby([COL_REGION, COL_YEAR])[COL_VALUE]
                .sum()
                .reset_index()
                .rename(columns={COL_VALUE: "bev_sales"})
            )
            cs_agg = (
                df_cs.groupby([COL_REGION, COL_YEAR])[COL_VALUE]
                .sum()
                .reset_index()
                .rename(columns={COL_VALUE: "charging_stations"})
            )
            corr_df = bev_agg.merge(cs_agg, on=[COL_REGION, COL_YEAR], how="inner")

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(
                corr_df["charging_stations"],
                corr_df["bev_sales"],
                alpha=0.4,
                color="steelblue",
                edgecolors="none",
            )
            ax.set_title("Korelacja: stacje ladowania vs sprzedaz BEV (per kraj-rok)")
            ax.set_xlabel("Liczba stacji ladowania")
            ax.set_ylabel("Sprzedaz BEV")
            plt.tight_layout()
            plt.show()

            r = corr_df[["charging_stations", "bev_sales"]].corr().iloc[0, 1]
            print(f"Korelacja Pearsona (stacje ladowania vs sprzedaz BEV): {r:.4f}")

    # 6. Correlation heatmap (numeric columns of BEV long data)
    numeric_cols = df_bev.select_dtypes(include="number")
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Macierz korelacji (dane BEV)")
    plt.tight_layout()
    plt.show()
