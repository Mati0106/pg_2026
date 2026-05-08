"""Visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

from src.settings import CHARTS_DIR, CORRELATION_DIR, EDA_DIR, PROJECT_ROOT, SHAP_DIR


def _plot_correlation_heatmap(
    df: pd.DataFrame,
    method: str,
    title: str,
    filename: str,
) -> None:
    """Plot correlation heatmap for Churn column."""
    correlation = df.corr(method=method)
    plt.figure(figsize=(15, 10))
    sns.heatmap(
        correlation[["Churn"]].sort_values(by="Churn", ascending=False),
        annot=True,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
    )
    plt.title(title, pad=20)
    plt.tight_layout()
    CORRELATION_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CORRELATION_DIR / filename, bbox_inches="tight", dpi=300)


def plot_pearson_correlation(df: pd.DataFrame) -> None:
    """Plot Pearson correlation heatmap for Churn."""
    _plot_correlation_heatmap(
        df,
        method="pearson",
        title="What affects customer churn (Churn)?",
        filename="churn_correlation_pearson.png",
    )


def plot_spearman_correlation(df: pd.DataFrame) -> None:
    """Plot Spearman correlation heatmap for Churn."""
    _plot_correlation_heatmap(
        df,
        method="spearman",
        title="What affects customer churn (Spearman)?",
        filename="churn_correlation_spearman.png",
    )


def plot_kendall_correlation(df: pd.DataFrame) -> None:
    """Plot Kendall correlation heatmap for Churn."""
    _plot_correlation_heatmap(
        df,
        method="kendall",
        title="What affects customer churn (Kendall)?",
        filename="churn_correlation_kendall.png",
    )


def visualize_feature_relationships(df: pd.DataFrame) -> None:
    """Plot correlation heatmaps and Mutual Information for features vs Churn."""
    plot_correlation_heatmaps(df)
    plot_mutual_information(df)


def plot_correlation_heatmaps(df: pd.DataFrame) -> None:
    """Plot Pearson, Spearman, and Kendall correlation heatmaps for Churn."""
    plot_pearson_correlation(df)
    plot_spearman_correlation(df)
    plot_kendall_correlation(df)


def plot_mutual_information(df: pd.DataFrame) -> None:
    """Plot Mutual Information scores for features vs Churn target."""
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_scores_series = pd.Series(mi_scores, index=X.columns)
    mi_scores_series = mi_scores_series.sort_values(ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=mi_scores_series.values, y=mi_scores_series.index, hue=mi_scores_series.index, palette="viridis", legend=False)

    plt.title("What affects customer churn (Mutual Information)", pad=20)
    plt.xlabel("Mutual Information score (higher = more important feature)")
    plt.ylabel("Features (Columns)")

    plt.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHARTS_DIR / "mutual_information.png", bbox_inches="tight", dpi=300)
    plt.close()

    print("\nExact Mutual Information results:")
    print(mi_scores_series)
    print(
        "None of the standard correlations (Pearson, Spearman, Kendall) are ideal "
        "for a categorical target. Mutual Information is based on Entropy, so it "
        "can catch non-linear and much more complex dependencies."
    )


def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Plot and save the Confusion Matrix for the model."""
    # Generate a 2x2 grid comparing actual outcomes vs model predictions
    cm = confusion_matrix(y_true, y_pred)
    # Flatten the 2x2 matrix into a single line and unpack into separate variables
    tn, fp, fn, tp = cm.ravel()

    total = len(y_true)
    print(f"\nFrom {total} test cases:")
    print(f"  TN (True Negative)  = {tn}  – correctly predicted Stays")
    print(f"  FP (False Positive) = {fp}  – wrongly predicted Churns (actually Stays)")
    print(f"  FN (False Negative) = {fn}  – wrongly predicted Stays (actually Churns)")
    print(f"  TP (True Positive)  = {tp}  – correctly predicted Churns")
    print(cm)

    annotations = [
        [f"TN – True Negative\n{cm[0, 0]} ({cm[0, 0] / total * 100:.1f}%)\nCorrectly predicted Stays", f"FP – False Positive\n{cm[0, 1]} ({cm[0, 1] / total * 100:.1f}%)\nWrongly predicted Churns"],
        [f"FN – False Negative\n{cm[1, 0]} ({cm[1, 0] / total * 100:.1f}%)\nWrongly predicted Stays", f"TP – True Positive\n{cm[1, 1]} ({cm[1, 1] / total * 100:.1f}%)\nCorrectly predicted Churns"],
    ]

    # Define business colors (Hex)
    color_tn = "#A5D6A7"  # 0: Light green  (TN)
    color_fp = "#FFE082"  # 1: Light yellow  (FP)
    color_fn = "#EF9A9A"  # 2: Light red     (FN)
    color_tp = "#A5D6A7"  # 3: Light green   (TP)

    # Create a palette with exactly these 4 colors in the correct order
    custom_cmap = ListedColormap([color_tn, color_fp, color_fn, color_tp])
    
    # Create an index matrix so Seaborn knows which color to use where
    color_indices = [[0, 1],
                     [2, 3]]

    plt.figure(figsize=(8, 6))

    # Draw heatmap with custom colors
    sns.heatmap(
        color_indices,       
        annot=annotations,
        fmt="",
        cmap=custom_cmap,    
        cbar=False,
        xticklabels=["Stays (0)", "Churns (1)"],
        yticklabels=["Stays (0)", "Churns (1)"],
        annot_kws={"size": 13, "weight": "bold"}
    )

    plt.title(title, pad=20, fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")

    plt.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHARTS_DIR / filename, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"\nConfusion matrix chart saved as 'charts/{filename}'")


def _plot_contract_churn(df: pd.DataFrame) -> None:
    """Bar chart: Churn by Contract Type."""
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Contract", hue="Churn", palette="Blues")
    plt.title("Churn by Contract Type", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(EDA_DIR / "eda_1_contract.png", bbox_inches="tight", dpi=300)
    plt.close()


def _plot_internet_churn(df: pd.DataFrame) -> None:
    """Bar chart: Churn by Internet Service."""
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="InternetService", hue="Churn", palette="Blues")
    plt.title("Churn by Internet Service", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(EDA_DIR / "eda_2_internet.png", bbox_inches="tight", dpi=300)
    plt.close()


def _plot_tenure_churn(df: pd.DataFrame) -> None:
    """Boxplot: Tenure vs Churn (Customer Loyalty)."""
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Churn", y="tenure", palette="Blues", hue="Churn", legend=False)
    plt.title("Tenure vs Churn (Customer Loyalty)", fontsize=14, fontweight="bold")
    plt.ylabel("Tenure (Months)")
    plt.tight_layout()
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(EDA_DIR / "eda_3_tenure.png", bbox_inches="tight", dpi=300)
    plt.close()


def _plot_charges_churn(df: pd.DataFrame) -> None:
    """Boxplot: Monthly Charges vs Churn."""
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", palette="Blues", hue="Churn", legend=False)
    plt.title("Monthly Charges vs Churn", fontsize=14, fontweight="bold")
    plt.ylabel("Monthly Charges (USD)")
    plt.tight_layout()
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(EDA_DIR / "eda_4_charges.png", bbox_inches="tight", dpi=300)
    plt.close()


def _plot_payment_churn(df: pd.DataFrame) -> None:
    """Bar chart: Churn by Payment Method."""
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x="PaymentMethod", hue="Churn", palette="Blues")
    plt.title("Churn by Payment Method", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=15)
    plt.tight_layout()
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(EDA_DIR / "eda_5_payment.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_business_insights(df: pd.DataFrame) -> None:
    """Generate 5 business charts showing the root causes of churn."""
    print("\nGenerating business charts (Root Cause Analysis)...")
    sns.set_theme(style="whitegrid")

    _plot_contract_churn(df)
    _plot_internet_churn(df)
    _plot_tenure_churn(df)
    _plot_charges_churn(df)
    _plot_payment_churn(df)

    print(f"Saved 5 charts in the '{EDA_DIR.relative_to(PROJECT_ROOT)}' folder")


def plot_shap_importance(xgb_model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Generate SHAP feature importance charts for the tuned XGBoost model.

    Saves 4 charts to charts/:
    - shap_summary.png         — global feature importance (beeswarm)
    - shap_dependence_top1.png — dependence plot for the most important feature
    - shap_dependence_top2.png — dependence plot for the second most important feature
    - shap_force.png           — local explanation for a single churner
    - shap_waterfall.png       — waterfall breakdown for the same churner
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    # Reset seaborn theme set by plot_business_insights — SHAP plots should have no grid
    sns.reset_orig()

    explainer   = shap.Explainer(xgb_model, X_test)
    shap_values = explainer(X_test)

    feature_names = X_test.columns.tolist()

    # --- 1. Summary Plot — global feature importance ---
    plt.figure(figsize=(14, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False, plot_size=None)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_summary.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved: charts/shap/shap_summary.png")

    # --- 2. Dependence Plots — top 2 most important features ---
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    top2_features = pd.Series(mean_shap, index=feature_names).nlargest(2).index.tolist()

    for i, feature in enumerate(top2_features, start=1):
        plt.figure()
        shap.dependence_plot(feature, shap_values.values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(SHAP_DIR / f"shap_dependence_top{i}.png", bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved: charts/shap/shap_dependence_top{i}.png  (feature: {feature})")

    # --- 3. Force Plot + Waterfall — local explanation for first churner in test set ---
    y_test_reset  = y_test.reset_index(drop=True)
    churner_idx   = y_test_reset[y_test_reset == 1].index[0]

    # Force Plot
    plt.figure()
    shap.force_plot(
        explainer.expected_value,
        shap_values[churner_idx].values,
        X_test.iloc[churner_idx],
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_force.png", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: charts/shap/shap_force.png  (churner index: {churner_idx})")

    # Waterfall Plot — SHAP creates its own figure internally, resize it after the fact
    shap.plots.waterfall(shap_values[churner_idx], show=False)
    plt.gcf().set_size_inches(14, 8)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_waterfall.png", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: charts/shap/shap_waterfall.png  (churner index: {churner_idx})")

