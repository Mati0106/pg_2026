from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

DATASET = "padmapiyush/global-electric-vehicle-dataset-2023"
RAW_PATH = "src/scripts/data/raw.csv"
PROCESSED_PATH = "src/scripts/data/processed.csv"
REGION_MAP_PATH = "src/scripts/data/region_map.csv"

# raw dataset column names
COL_REGION = "region"
COL_YEAR = "year"
COL_VALUE = "value"
COL_PARAMETER = "parameter"
COL_POWERTRAIN = "powertrain"

# column names after reshape (wide format)
COL_EV_SALES = "ev_sales"
COL_EV_STOCK = "ev_stock"
COL_CHARGING = "charging_stations"

BEV = "BEV"
PARAM_SALES = "EV sales"
PARAM_STOCK = "EV stock"
PARAM_CHARGING = "EV charging points"


def load_data():
    df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, DATASET, "")
    print("Pobrano dane:", df.shape)
    print("Kolumny:", list(df.columns))
    if COL_PARAMETER in df.columns:
        print("Parametry:", sorted(df[COL_PARAMETER].unique().tolist()))
    os.makedirs("src/scripts/data", exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    return df


def filter_bev(df):
    if COL_POWERTRAIN in df.columns:
        df = df[df[COL_POWERTRAIN] == BEV].copy()
        print("Po filtrze BEV:", df.shape)
    else:
        print("Brak kolumny powertrain - pomijam filtr BEV")
    return df


def extract_charging(df_raw):
    """Aggregate total charging stations per region-year from unfiltered raw data."""
    df_cs = df_raw[df_raw[COL_PARAMETER] == PARAM_CHARGING].copy()
    if df_cs.empty:
        print("UWAGA: brak danych o stacjach ladowania w zbiorze")
        return pd.DataFrame(columns=[COL_REGION, COL_YEAR, COL_CHARGING])
    df_cs = (
        df_cs.groupby([COL_REGION, COL_YEAR])[COL_VALUE]
        .sum()
        .reset_index()
        .rename(columns={COL_VALUE: COL_CHARGING})
    )
    print("Stacje ladowania - unikalne regiony:", df_cs[COL_REGION].nunique())
    return df_cs


def run():
    df_raw = pd.read_csv(RAW_PATH)

    # extract charging stations BEFORE BEV filter (not BEV-specific infrastructure)
    df_charging = extract_charging(df_raw)

    # filter to BEV and historical years only
    df = df_raw.copy()
    if COL_POWERTRAIN in df.columns:
        df = df[df[COL_POWERTRAIN] == BEV].copy()
    df = df[df[COL_YEAR] <= 2023]

    # clean
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=[COL_REGION, COL_YEAR, COL_VALUE])
    print(f"Usunieto {before - len(df)} wierszy (duplikaty / braki)")

    # reshape long -> wide: one row per (region, year)
    df_s = (
        df[df[COL_PARAMETER] == PARAM_SALES][[COL_REGION, COL_YEAR, COL_VALUE]]
        .rename(columns={COL_VALUE: COL_EV_SALES})
    )
    df_st = (
        df[df[COL_PARAMETER] == PARAM_STOCK][[COL_REGION, COL_YEAR, COL_VALUE]]
        .rename(columns={COL_VALUE: COL_EV_STOCK})
    )
    df_wide = df_s.merge(df_st, on=[COL_REGION, COL_YEAR], how="inner")

    # merge charging stations (left join — not every region has data)
    df_charging_hist = df_charging[df_charging[COL_YEAR] <= 2023]
    df_wide = df_wide.merge(df_charging_hist, on=[COL_REGION, COL_YEAR], how="left")

    # forward-fill missing charging values per region, then fill remaining with 0
    df_wide = df_wide.sort_values([COL_REGION, COL_YEAR])
    df_wide[COL_CHARGING] = (
        df_wide.groupby(COL_REGION)[COL_CHARGING]
        .transform(lambda s: s.ffill().bfill())
        .fillna(0)
    )

    # target: BEV sales as % of total stock; cap at 100% (data quality guard)
    df_wide["ev_adoption_rate"] = df_wide[COL_EV_SALES] / df_wide[COL_EV_STOCK] * 100
    df_wide = df_wide[df_wide["ev_adoption_rate"] <= 100].copy()

    # year-over-year growth per region
    df_wide = df_wide.sort_values([COL_REGION, COL_YEAR])
    df_wide["yoy_growth"] = (
        df_wide.groupby(COL_REGION)[COL_EV_SALES].pct_change() * 100
    )

    # save region name -> numeric code mapping (needed for forecast)
    region_categories = sorted(df_wide[COL_REGION].unique())
    region_to_code = {r: i for i, r in enumerate(region_categories)}
    pd.DataFrame(
        list(region_to_code.items()), columns=["region", "code"]
    ).to_csv(REGION_MAP_PATH, index=False)

    # encode region as numeric
    df_wide[COL_REGION] = df_wide[COL_REGION].astype("category").cat.codes

    df_clean = df_wide.dropna(subset=["ev_adoption_rate", "yoy_growth"]).copy()
    df_clean.to_csv(PROCESSED_PATH, index=False)
    print("Przetworzone dane:", df_clean.shape)
    print(df_clean["ev_adoption_rate"].describe().round(2))
    return df_clean
