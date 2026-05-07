from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from users.kczap.profiling_data.load_honey_dataset import load_honey_data

# Load dataset
dataset = load_honey_data()

# One-hot encode
df_encoded = pd.get_dummies(dataset, columns=["Pollen_analysis"], drop_first=True)

X = df_encoded.drop(["Purity","Price"], axis=1)
Y = df_encoded["Purity"]

seed = 7
test_size = 0.33
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# Initialize a basic XGBoost regression model + Train the model
model = XGBRegressor()
model.fit(X_train, y_train)


print(model)

# Predict
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("MSE: {0}".format(mse))


# importances = model.feature_importances_
# importance_df = pd.DataFrame({
#     "feature": X.columns,
#     "importance": importances
# }).sort_values(by="importance", ascending=False)
#
# print("\nFeature Importances:")
# print(importance_df)
#
#
# plt.figure(figsize=(10, 8))
# plot_importance(model, max_num_features=20)
# plt.title("XGBoost Feature Importance")
# plt.show()


#  Optuna
best_params = {
    'max_depth': 8,
    'learning_rate': 0.07369811109388126,
    'n_estimators': 341,
    'subsample': 0.7163224565957149,
    'colsample_bytree': 0.795826749579538,
    'gamma': 0.004807135534631489,
    'min_child_weight': 1

}

final_model = XGBRegressor(
    objective='reg:squarederror',
    **best_params)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Final Model MSE:", mse)
