"""Utility functions."""

import ast
import pandas as pd

from src.settings import DATA_DIR


def save_to_csv(df: pd.DataFrame, filename: str, index: bool = False) -> None:
    """Save DataFrame to CSV file in the data folder."""
    path = DATA_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def get_best_params(filename: str, model_name: str) -> dict:
    """Read the best hyperparameters for a given model from a CSV results file."""
    df = pd.read_csv(DATA_DIR / filename)
    best_params_str = df.loc[df["Tuned Model"] == model_name, "Best Params"].values[0]
    # ast.literal_eval safely converts a string like "{'C': 0.1, 'max_depth': 3}" back to a dict
    return ast.literal_eval(best_params_str)

