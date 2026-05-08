"""Project constants and configuration."""

from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # __file__ is not defined in interactive sessions (e.g. IPython/REPL)
    PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"
CHARTS_DIR = PROJECT_ROOT / "charts"
EDA_DIR = CHARTS_DIR / "eda"
CORRELATION_DIR = CHARTS_DIR / "correlation"
SHAP_DIR = CHARTS_DIR / "shap"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATASET_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
KAGGLE_DATASET = "blastchar/telco-customer-churn"
REPORT_FILENAME = "Telco_Customer_Churn_report.html"

settings = {
    "binary_cols": [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
    ],
    "one_hot_encoding_cols": [
        "PaymentMethod",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "MultipleLines"
    ],
}
