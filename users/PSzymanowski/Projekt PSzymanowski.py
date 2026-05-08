import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from xgboost import plot_importance

import shap

# Data collection

dataset = pd.read_csv('social_media_productivity_6000.csv')

print(dataset.head())
print(dataset.info())
print(dataset.describe())
print(dataset.isnull().sum())

# Feature engineering + Engineering analysis

data = dataset.copy()

numeric_columns = data.select_dtypes(include=np.number).columns

for col in numeric_columns:
    data[col] = data[col].fillna(data[col].median())

categorical_columns = data.select_dtypes(include='object').columns

for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

data = data.dropna(subset=['productivity_score'])

X = data.drop('productivity_score', axis=1)
y = data['productivity_score']

X = pd.get_dummies(X, drop_first=True)

correlation_data = X.copy()
correlation_data['productivity_score'] = y

print(correlation_data.corr()['productivity_score'].sort_values(ascending=False))

# Modeling

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=7
)

model = RandomForestRegressor(random_state=7)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print('Random Forest MSE: %.3f' % mse)
print('Random Forest MAE: %.3f' % mae)

model = XGBRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print('XGBoost MSE: %.3f' % mse)
print('XGBoost MAE: %.3f' % mae)

# Hiperparametry optymalizacyjne

grid = dict()

grid['max_depth'] = [3, 5, 7]
grid['n_estimators'] = [50, 100]
grid['learning_rate'] = [0.01, 0.1]

search = GridSearchCV(
    XGBRegressor(),
    grid,
    scoring='neg_mean_squared_error',
    cv=3,
    n_jobs=-1
)

results = search.fit(X_train, y_train)

print('Best score: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)

best_model = results.best_estimator_

best_predictions = best_model.predict(X_test)

best_mse = mean_squared_error(y_test, best_predictions)
best_mae = mean_absolute_error(y_test, best_predictions)

print('Optimized XGBoost MSE: %.3f' % best_mse)
print('Optimized XGBoost MAE: %.3f' % best_mae)

# Interpretacja wyników

print("Feature importances {0}".format(best_model.feature_importances_))

plot_importance(best_model)
plt.show()

explainer = shap.TreeExplainer(best_model)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)
