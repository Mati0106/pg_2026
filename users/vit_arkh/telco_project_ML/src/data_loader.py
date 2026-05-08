"""Data loading utilities."""

import pandas as pd
from dotenv import load_dotenv

from src.settings import DATA_DIR, DATASET_FILENAME, KAGGLE_DATASET, PROJECT_ROOT

# Load .env BEFORE importing kaggle so credentials are in os.environ when kaggle initializes
load_dotenv(PROJECT_ROOT / ".env")

import kaggle  # noqa: E402


def download_dataset() -> None:
    """Download the Telco Customer Churn dataset from Kaggle to the data folder."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    kaggle.api.dataset_download_files(
        KAGGLE_DATASET, path=str(DATA_DIR), unzip=True
    )


def load_dataset() -> pd.DataFrame:
    """Load the Telco Customer Churn dataset from the data folder."""
    return pd.read_csv(DATA_DIR / DATASET_FILENAME)


def load_data() -> pd.DataFrame:
    """Download dataset from Kaggle (if needed) and load it into a DataFrame."""
    download_dataset()
    return load_dataset()
