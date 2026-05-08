from models_utils import *


X, y = load_and_prepare_data()
X_train, X_test, y_train, y_test = split_data(X, y)
# #  Optuna
best_params = {
    'max_depth': 8,
    'learning_rate': 0.07369811109388126,
    'n_estimators': 341,
    'subsample': 0.7163224565957149,
    'colsample_bytree': 0.795826749579538,
    'gamma': 0.004807135534631489,
    'min_child_weight': 1

}
baseline_model = train_baseline_model(X_train, y_train)
baseline_metrics = evaluate_model(baseline_model, X_test, y_test)
print("Baseline:", baseline_metrics)

final_model = train_final_model(X_train, y_train, best_params)
final_metrics = evaluate_model(final_model, X_test, y_test)
print("Final:", final_metrics)

plot_shap_all(
    model=final_model,
    X_train=X_train,
    X_test=X_test,
    feature_names=X.columns,
    feature="Density",
    index=0
)


# Logistic Regression

# Load classification data
X, y = load_and_prepare_classification_data()

# Split
X_train, X_test, y_train, y_test = split_data(X, y)

# Train logistic regression
log_model = train_logistic_model(X_train, y_train)

# Evaluate
accuracy, report = evaluate_classification_model(log_model, X_test, y_test)

print("Logistic Regression Accuracy: %.2f%%" % (accuracy * 100))
print(report)