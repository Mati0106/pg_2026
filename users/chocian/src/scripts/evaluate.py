import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import PartialDependenceDisplay


def run():
    model = joblib.load("src/modelling/xgb_model.pkl")
    X_test, y_test = joblib.load("src/modelling/test_data.pkl")

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print("RMSE: %.4f | MAE: %.4f | R2: %.4f" % (rmse, mae, r2))

    # feature importance
    fi = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False)
    fi.plot(kind="barh", title="Feature Importance - XGBoost")
    plt.tight_layout()
    plt.show()

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=True)

    # PDP top 3 features
    top3 = fi.head(3).index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    PartialDependenceDisplay.from_estimator(model, X_test.astype(float), top3, ax=axes)
    plt.suptitle("Partial Dependence Plots")
    plt.tight_layout()
    plt.show()
