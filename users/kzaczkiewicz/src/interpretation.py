# external
import matplotlib
matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt
import shap
import pandas as pd

# --- CONSTANTS ---
PLOT_FIGSIZE = (10, 6)
TOP_N_FEATURES = 10


def feature_importance_plot(model, feature_names: list, save_path: str = None) -> None:
    """Bar chart of XGBoost feature importances (gain). Saves or displays the plot."""
    importances = model.feature_importances_
    importance_df = (
        pd.Series(importances, index=feature_names)
        .sort_values(ascending=True)
        .tail(TOP_N_FEATURES)
    )

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    importance_df.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("XGBoost Feature Importances (Gain)")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    # Conclusion printed below the chart
    top_feature = importance_df.index[-1]
    print(
        f"\n[WNIOSEK] Najważniejszy feature to '{top_feature}'. "
        "Cechy z wysoką wartością gain najbardziej wpływają na predykcję naprawy buga."
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def shap_analysis(model, X_test: pd.DataFrame, save_path: str = None) -> None:
    """SHAP beeswarm summary plot showing direction and magnitude of each feature's impact."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=PLOT_FIGSIZE)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary Plot — wpływ featurów na predykcję")

    print(
        "\n[WNIOSEK] Wykres SHAP pokazuje kierunek wpływu każdej cechy: "
        "czerwone punkty (wysoka wartość featura) przesunięte w prawo zwiększają "
        "prawdopodobieństwo klasyfikacji jako 'Fixed'."
    )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
