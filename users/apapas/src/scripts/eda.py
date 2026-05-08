import glob

import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


def clean_column(col):
    col = col.astype(str)
    col = col.str.replace("$", "", regex=False)
    col = col.str.replace(",", "", regex=False)
    col = col.str.replace("%", "", regex=False)
    col = col.str.replace("-", "", regex=False)
    col = col.str.strip()
    col = col.replace("", float("nan"))
    return pd.to_numeric(col, errors="coerce")


def plot_histogram(ax, data, col):
    ax.hist(data[col].dropna(), bins=30, color="blue", edgecolor="white", alpha=0.8)
    ax.set_title(col)
    ax.set_ylabel("Liczba firm")
    ax.tick_params(axis="x", rotation=45)


def count_iqr_outliers(series):
    series = series.dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    is_outlier = (series < lower) | (series > upper)
    return is_outlier.sum()


def run_eda():
    path = kagglehub.dataset_download("mirzayasirabdullah07/fortune-500-companies-us")
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    df = pd.read_csv(csv_files[0], encoding="latin1")
    print(df.shape)
    print(df.head())

    # czyszczenie kolumn - usuwanie znakow specjalnych
    cols_to_clean = [
        "Number of Employees",
        "Previous Rank",
        "Revenues",
        "Revenue Change",
        "Profits",
        "Profit Change",
        "Assets",
        "Market Value",
    ]

    for col in cols_to_clean:
        df[col] = clean_column(df[col])

    print("\nBraki:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    print("\nStatystyki:")
    print(df.describe().round(2))

    os.makedirs("wyniki", exist_ok=True)

    # histogramy
    cols = ["Revenues", "Profits", "Assets", "Market Value", "Number of Employees"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Rozklady zmiennych num", fontsize=16, fontweight="bold")

    ax_list = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]
    for ax, col in zip(ax_list, cols):
        plot_histogram(ax, df, col)

    axes[1, 2].set_visible(False)
    plt.tight_layout()
    plt.savefig("wyniki/histogramy.png", dpi=150)
    plt.close()

    # wnioski z histogramow
    print(f"\nSkosnosc Market Value: {df['Market Value'].dropna().skew():.2f}")
    print(f"Skosnosc Revenues: {df['Revenues'].dropna().skew():.2f}")

    # macierz korelacji
    corr_cols = [
        "Rank",
        "Revenues",
        "Profits",
        "Assets",
        "Market Value",
        "Number of Employees",
    ]
    corr = df[corr_cols].corr().round(2)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, ax=ax
    )
    ax.set_title("Macierz korelacji")
    plt.tight_layout()
    plt.savefig("wyniki/macierz_korelacji.png", dpi=150)
    plt.close()

    print(f"\nKorelacje z Market Value:")
    print(f"  Profits: {corr.loc['Profits', 'Market Value']}")
    print(f"  Revenues: {corr.loc['Revenues', 'Market Value']}")
    print(f"  Assets: {corr.loc['Assets', 'Market Value']}")

    # scatter ploty
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(df["Revenues"], df["Market Value"], alpha=0.5, color="blue", s=20)
    axes[0].set_xlabel("Revenues (mln $)")
    axes[0].set_ylabel("Market Value (mln $)")
    axes[0].set_title("Revenues vs Market Value")

    axes[1].scatter(df["Profits"], df["Market Value"], alpha=0.5, color="brown", s=20)
    axes[1].set_xlabel("Profits (mln $)")
    axes[1].set_ylabel("Market Value (mln $)")
    axes[1].set_title("Profits vs Market Value")

    axes[2].scatter(df["Assets"], df["Market Value"], alpha=0.5, color="green", s=20)
    axes[2].set_xlabel("Assets (mln $)")
    axes[2].set_ylabel("Market Value (mln $)")
    axes[2].set_title("Assets vs Market Value")

    plt.suptitle("Zaleznosci zmiennych z Market Value", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("wyniki/scatter_plots.png", dpi=150)
    plt.close()

    # boxploty
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for i, col in enumerate(["Revenues", "Profits", "Assets", "Market Value"]):
        axes[i].boxplot(df[col].dropna(), vert=True)
        axes[i].set_title(col)
        axes[i].set_ylabel("mln $")

    plt.suptitle("Identyfikacja outlinerow", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("wyniki/boxploty.png", dpi=150)
    plt.close()

    # ile outlinerow w MV?
    n_outliers = count_iqr_outliers(df["Market Value"])
    print(f"\nMarket Value ma {n_outliers} outlinerow (IQR)")

    print("\nWykresy zapisane w wyniki/")
    return df


if __name__ == "__main__":
    run_eda()
