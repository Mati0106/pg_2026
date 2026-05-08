# external
from dotenv import load_dotenv
import os
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# internal
from src.preprocessing import (
    FEATURE_COLS,
    TARGET_COL,
    load_data,
    feature_engineering,
    split_and_scale,
)
from src.modeling import modeling, optimize_with_optuna, RANDOM_STATE
from src.interpretation import feature_importance_plot, shap_analysis

# --- CONSTANTS ---
SAVE_PLOTS = True # If True, saves charts as files instead of opening them in a popup window


load_dotenv()

DATA_PATH = os.getenv('DATA_PATH')

def main():
    # Load raw Jira data from CSV file.
    print("=== 1. Loading data ===")
    df_raw = load_data(DATA_PATH)
    print(f"Resolved issues loaded: {len(df_raw)}")

    # Clean data and transform text into numerical features.
    print("\n=== 2. Feature engineering ===")
    df = feature_engineering(df_raw)
    print(f"Features built: {FEATURE_COLS}")
    print(f"Target distribution:\n{df[TARGET_COL].value_counts()}")

    # Split data into training (70%) and testing (30%) sets. Normalize values.
    # Scale data AFTER split to avoid data leakage.
    print("\n=== 3. Train/test split (70/30) + StandardScaler ===")
    X_train_s, X_test_s, X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Train a baseline Logistic Regression and a simple XGBoost for comparison.
    print("\n=== 4. Modeling ===")
    results = modeling(X_train_s, X_test_s, X_train, X_test, y_train, y_test)

    for name, res in results.items():
        print(f"\n--- {name} ---")
        print(f"  Accuracy : {res['accuracy']:.4f}")
        print(f"  AUC-ROC  : {res['auc']:.4f}")
        print(res["report"])

    # Use Optuna to find the most efficient hyperparameters for the XGBoost model.
    print("\n=== 5. Optuna hyperparameter optimization ===")
    best_params = optimize_with_optuna(X_train, y_train)

    # Train the final XGBoost model using the best parameters found by Optuna.
    print("\n=== 6. Final XGBoost (Optuna-optimized) ===")
    final_model = xgb.XGBClassifier(
        **best_params,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    final_model.fit(X_train, y_train)
    final_pred = final_model.predict(X_test)
    final_proba = final_model.predict_proba(X_test)[:, 1]

    print(f"  Accuracy : {accuracy_score(y_test, final_pred):.4f}")
    print(f"  AUC-ROC  : {roc_auc_score(y_test, final_proba):.4f}")
    print(classification_report(y_test, final_pred))

    # Generate a plot showing which features had the most impact on the model.
    print("\n=== 7. Feature importance ===")
    feature_importance_plot(
        final_model,
        FEATURE_COLS,
        save_path="feature_importance.png" if SAVE_PLOTS else None,
    )

    # Perform SHAP analysis to explain individual predictions.
    print("\n=== 8. SHAP analysis ===")
    shap_analysis(
        final_model,
        X_test,
        save_path="shap_summary.png" if SAVE_PLOTS else None,
    )

    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    main()