from pathlib import Path
import pandas as pd


def load_honey_data(filename="honey_purity_dataset.csv"):
    base = Path(__file__).resolve().parent.parent
    path = base / "datasets" / filename

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    print("Sample of the data")
    print(df.head())
    return df
