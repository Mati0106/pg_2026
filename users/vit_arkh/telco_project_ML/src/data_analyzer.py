"""Data analysis utilities."""

import pandas as pd
from ydata_profiling import ProfileReport

from src.settings import REPORT_FILENAME, REPORTS_DIR


def generate_report(
        df: pd.DataFrame, report_name: str = REPORT_FILENAME
) -> None:
    """Generate a ydata-profiling HTML report for the dataset. Saves to reports/ folder."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / report_name

    report = ProfileReport(
        df,
        title="Telco Customer Churn Data",
        correlations={
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "kendall": {"calculate": True},
        },
    )
    report.to_file(str(output_path))
    print("\n'Telco Customer Churn' report was generated and saved to:", output_path)
