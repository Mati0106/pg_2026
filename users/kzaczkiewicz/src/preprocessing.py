# external
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# internal
# (no internal imports yet)

# --- CONSTANTS ---
TEST_SIZE = 0.30
RANDOM_STATE = 42
TARGET_COL = "was_fixed"

PRIORITY_MAP = {"Low": 0, "Medium": 1, "High": 2, "Highest": 3}
ISSUE_TYPE_MAP = {"Bug": 0, "Suggestion": 1}

FEATURE_COLS = [
    "priority_encoded",
    "issue_type_encoded",
    "votes",
    "description_length",
    "has_labels",
    "reporter_issue_count",
    "time_to_first_comment_days",
    "comment_count",
    "has_attachment",
]


def load_data(path: str) -> pd.DataFrame:
    """Load CSV from path and filter to resolved issues only."""
    df = pd.read_csv(path, low_memory=False)
    df = df[df["Resolution"].notna()].reset_index(drop=True)
    return df


def _parse_comment_date(comment_val) -> pd.Timestamp | None:
    """Extract timestamp from first Comment cell (format: 'DD/Mon/YYYY HH:MM AM;user;text')."""
    if pd.isna(comment_val):
        return None
    try:
        date_str = str(comment_val).split(";")[0].strip()
        return pd.to_datetime(date_str, format="%d/%b/%Y %I:%M %p", errors="coerce")
    except Exception:
        return None


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Build all model features and target variable from raw DataFrame."""
    out = pd.DataFrame()

    # priority_encoded
    out["priority_encoded"] = df["Priority"].map(PRIORITY_MAP).fillna(0).astype(int)

    # issue_type_encoded
    out["issue_type_encoded"] = (
        df["Issue Type"].map(ISSUE_TYPE_MAP).fillna(0).astype(int)
    )

    # votes
    out["votes"] = pd.to_numeric(df["Votes"], errors="coerce").fillna(0).astype(int)

    # description_length
    out["description_length"] = df["Description"].fillna("").str.len()

    # has_labels — any of the Labels columns is non-null
    label_cols = [c for c in df.columns if c.startswith("Labels")]
    out["has_labels"] = df[label_cols].notna().any(axis=1).astype(int)

    # reporter_issue_count — activity of the reporter
    reporter_counts = df["Reporter"].value_counts()
    out["reporter_issue_count"] = df["Reporter"].map(reporter_counts).fillna(1).astype(int)

    # time_to_first_comment_days
    created_dt = pd.to_datetime(df["Created"], format="%d/%b/%Y %I:%M %p", errors="coerce")
    comment_cols = [c for c in df.columns if c.startswith("Comment")]
    first_comment_dt = df[comment_cols[0]].apply(_parse_comment_date)
    diff = (first_comment_dt - created_dt).dt.days
    out["time_to_first_comment_days"] = diff.fillna(-1).astype(int)

    # comment_count — number of non-null Comment columns per row
    out["comment_count"] = df[comment_cols].notna().sum(axis=1).astype(int)

    # has_attachment
    attachment_cols = [c for c in df.columns if c.startswith("Attachment")]
    out["has_attachment"] = df[attachment_cols].notna().any(axis=1).astype(int)

    # target
    out[TARGET_COL] = (df["Resolution"] == "Fixed").astype(int)

    return out


def split_and_scale(df: pd.DataFrame):
    """Split 70/30, then fit StandardScaler on train only to avoid data leakage."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index
    )

    return X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, scaler
