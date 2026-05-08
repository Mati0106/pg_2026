import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import PartialDependenceDisplay
import os


def run_interpretation(model, X_test, features):
    os.makedirs("wyniki", exist_ok=True)

    # feature importance (drzewa: feature_importances_, modele liniowe: |coef_|)
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        importance = np.abs(model.coef_)
    df_imp = pd.DataFrame({"feature": features, "importance": importance})
    df_imp = df_imp.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_imp["feature"], df_imp["importance"], color="steelblue")
    ax.set_xlabel("Waznosc")
    ax.set_title("Feature Importance")
    plt.tight_layout()
    plt.savefig("wyniki/feature_importance.png", dpi=150)
    plt.close()

    print("Ranking cech:")
    for _, row in df_imp.sort_values("importance", ascending=False).iterrows():
        print(f"  {row['feature']:25s} {row['importance']:.4f}")

    top = df_imp.sort_values("importance", ascending=False).iloc[0]
    # tree models: feature_importances_ sums to 1 → percentage; linear models: raw |coef_|
    imp_fmt = f"{top['importance']:.1%}" if hasattr(model, "feature_importances_") else f"{top['importance']:.4f}"
    print(f"\nNajwazniejsza cecha: {top['feature']} ({imp_fmt})")

    # SHAP
    print("\nObliczanie SHAP...")
    explainer = shap.Explainer(model.predict, X_test)
    shap_values = explainer(X_test)

    # summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values.values, X_test, feature_names=features, show=False)
    plt.title("SHAP - wplyw cech na predykcje")
    plt.tight_layout()
    plt.savefig("wyniki/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values.values, X_test, feature_names=features,
                      plot_type="bar", show=False)
    plt.title("SHAP - sredni wplyw cech")
    plt.tight_layout()
    plt.savefig("wyniki/shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_shap = np.abs(shap_values.values).mean(axis=0)
    top_idx = np.argmax(mean_shap)
    print(f"Najwazniejsza cecha wg SHAP: {features[top_idx]} (sredni |SHAP| = {mean_shap[top_idx]:.0f})")

    # PDP dla top 4 cech
    top_4_idx = np.argsort(importance)[-4:][::-1]
    top_4 = [features[i] for i in top_4_idx]
    print(f"\nPDP dla: {top_4}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for feat, ax in zip(top_4, axes.ravel()):
        idx = features.index(feat)
        PartialDependenceDisplay.from_estimator(
            model, X_test, features=[idx],
            feature_names=features, ax=ax, kind="average"
        )
        ax.set_title(f"PDP: {feat}")

    plt.suptitle("Partial Dependence Plots", fontsize=14)
    plt.tight_layout()
    plt.savefig("wyniki/pdp_plots.png", dpi=150)
    plt.close()

    print("\nWykresy zapisane w wyniki/")
    return importance
