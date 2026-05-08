import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from xgboost import plot_importance

import shap

# Data collection

dataset = pd.read_csv('WLASCIWY_personality_dataset.csv')

print(dataset.head())
print(dataset.info())
print(dataset.describe())
print(dataset.isnull().sum())
print(dataset['Personality'].value_counts())

dataset['Personality'].value_counts().plot(kind='bar')
plt.title('Personality distribution')
plt.xlabel('Personality')
plt.ylabel('Count')
plt.show()


# Feature engineering + Engineering analysis

data = dataset.copy()

numeric_columns = data.select_dtypes(include=np.number).columns

for col in numeric_columns:
    data[col] = data[col].fillna(data[col].median())

categorical_columns = data.select_dtypes(include='object').columns

for col in categorical_columns:
    if col != 'Personality':
        data[col] = data[col].fillna(data[col].mode()[0])

data = data.dropna(subset=['Personality'])

mapping = {
    'Introvert': 0,
    'Extrovert': 1
}

data['Personality_encoded'] = data['Personality'].map(mapping)

data = data.dropna(subset=['Personality_encoded'])

X = data.drop(['Personality', 'Personality_encoded'], axis=1)
y = data['Personality_encoded']

X = pd.get_dummies(X, drop_first=True)


# Modeling

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=7
)


model = RandomForestClassifier(random_state=7)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print('Random Forest Accuracy: %.2f%%' % (accuracy * 100.0))


model = XGBClassifier(max_depth=5, gamma=10)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)

print('XGBoost Accuracy: %.2f%%' % (accuracy * 100.0))


# Hiperparametry optymalizacyjne

grid = dict()

grid['max_depth'] = [3, 5, 7]
grid['n_estimators'] = [50, 100]
grid['learning_rate'] = [0.01, 0.1]

search = GridSearchCV(
    XGBClassifier(),
    grid,
    scoring='accuracy',
    cv=3,
    n_jobs=-1
)

results = search.fit(X_train, y_train)

print('Best score: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)

best_model = results.best_estimator_

best_predictions = best_model.predict(X_test)
best_predictions = [round(value) for value in best_predictions]

best_accuracy = accuracy_score(y_test, best_predictions)

print('Optimized XGBoost Accuracy: %.2f%%' % (best_accuracy * 100.0))


# Interpretacja wyników

print("Feature importances {0}".format(best_model.feature_importances_))

plot_importance(best_model)
plt.show()

explainer = shap.TreeExplainer(best_model)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)