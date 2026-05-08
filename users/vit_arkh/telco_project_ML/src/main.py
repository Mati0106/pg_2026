"""
Telco Customer Churn - Machine Learning Analysis
Main entry point for the project.
ML Pipeline.
"""
#internal
import sys
from pathlib import Path

#external
import pandas as pd
pd.set_option("display.max_columns", None)

# Project root on path (so "from src.*" imports work for both run modes)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src.data_analyzer as data_analyzer
import src.data_loader as data_loader
import src.data_preprocessing as data_preprocessing
import src.visualization as visualization
import src.model_trainer as model_trainer


def main() -> None:
    """Main function - project entry point."""
    # 1. Download dataset
    df = data_loader.load_data()

    # 2. Generate HTML EDA report
    data_analyzer.generate_report(df)

    # 3. EDA, Business visualization - Generate charts visualizing churn dependency on different features
    visualization.plot_business_insights(df)

    # 4. Data preprocessing — encode everything (we don't know yet what's useful)
    df = data_preprocessing.clean_and_encode_data(df)
    # 5. Visualize correlation between features and target variable
    visualization.visualize_feature_relationships(df)

    # Model Training & Evaluation
    # 6. Train-test split
    X_train, X_test, y_train, y_test = model_trainer.split_data(df)

    # 7. Train and compare models — returns y_pred for Logistic Regression (baseline winner)
    y_pred_log_reg = model_trainer.train_and_compare_models(X_train, X_test, y_train, y_test)

    # 8. Confusion Matrix for Logistic Regression (winner)
    model_trainer.plot_logistic_regression_confusion_matrix(y_test, y_pred_log_reg)

    # 9. Tune hyperparameters for all models — returns tuned XGBoost model and its predictions
    xgb_tuned_model, y_pred_xgb_tuned = model_trainer.tune_hyperparameters(X_train, X_test, y_train, y_test)

    # 10. Confusion Matrix for tuned XGBoost (winner after tuning)
    model_trainer.plot_tuned_xgboost_confusion_matrix(y_test, y_pred_xgb_tuned)

    # 11. Business impact simulation comparing Logistic Regression and tuned XGBoost
    model_trainer.print_business_impact_simulation(y_test, y_pred_log_reg, y_pred_xgb_tuned)

    # 12. SHAP Feature Importance — which features drive XGBoost predictions
    visualization.plot_shap_importance(xgb_tuned_model, X_test, y_test)


if __name__ == "__main__":
    main()
