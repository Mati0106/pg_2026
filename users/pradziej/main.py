import logging as log
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.DataLoader import DataLoader
from src.DataConverter import DataConverter
from src.settings import settings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

log.basicConfig(level=log.INFO)

# 1. Read the data

data_loader = DataLoader()
df: pd.DataFrame = data_loader.load_data(settings['dataset'], force_download=False)

# 2. Data analysis + preprocessing
data_converter = DataConverter()

# Data understanding
data_understanding_explanation = """
Data has been collected from different hospitals, community clinics, maternal health cares through the IoT based risk monitoring system.

Columns:
 - Age - age of the person [int],
 - SystolicBP - Upper value of Blood Pressure in mmHg [float]
 - DiastolicBP - Lower value of Blood Pressure in mmHg [float]
 - BS - Blood glucose levels is in terms of a molar concentration, mmol/L [float]
 - BodyTemp - body temperature [F],
 - HeartRate - A normal resting heart rate in beats per minute [int].
 - RiskLevel - Predicted Risk Intensity Level during pregnancy -> THIS WILL BE OUR TARGET


 MAIN QUESTION:
  - Based on the simple measurements, that each future mum can make at home, provide a risk rank (high, medium, low)
  - Prepare a model, that can answer if the pregnancy is in danger
  - Just answer what is the most important parameter (from the available set) to measure during pregnancy.
"""

print(data_understanding_explanation)
print(f'Columns, {df.columns}')
print(f'RiskLevel:\n{df['RiskLevel'].unique()}')

print('Head of the data', df.head())
print('Tail of the data', df.tail())
print('Types of the columns', df.dtypes)
print('General info: ', df.info())

print('Amount of rows(observations), columns(features) [raw data]', df.shape)

df.drop_duplicates(inplace=True)
print('Amount of rows(observations), columns(features) [without duplicates data]', df.shape)

df.dropna(inplace=True)
print('Amount of rows(observations), columns(features) [without null values]', df.shape)

# Convert a subset of columns to categorical
RiskLevelConversionMap = {
    'low risk': 0,
    'mid risk': 1,
    'high risk': 2
}
data_converter.encode_categorial_with_map(df, 'RiskLevel', RiskLevelConversionMap)

# BodyTemp is in Fahrenheit - convert it to Celsius
data_converter.fahrenheit_to_celcius(df, 'BodyTemp')

# 3. Feature engineering
target_names = ['RiskLevel']
feature_names = [c for c in df.columns if c not in target_names]
print(f'Feature names: {feature_names}, Target names: {target_names}')
[print(f'RiskLevel correlation with {x}', df[['RiskLevel',x]].corr()) for x in feature_names]

"""
print('RiskLevel correlation with Age', df[['RiskLevel','Age']].corr())
print('RiskLevel correlation with SystolicBP', df[['RiskLevel','SystolicBP']].corr())
print('RiskLevel correlation with DiastolicBP', df[['RiskLevel','DiastolicBP']].corr())
print('RiskLevel correlation with BS', df[['RiskLevel','BS']].corr())
print('RiskLevel correlation with BS', df[['RiskLevel','BodyTemp']].corr())
print('RiskLevel correlation with HeartRate', df[['RiskLevel','HeartRate']].corr())
"""

X, y = df[feature_names], df[target_names]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=settings['test_size'], random_state=settings['random_state'])

# Shape - shows amount of rows and columns (works like list length, but for matrix)
print(f'Training Shape X:', X_train.shape, 'Testing Shape X:', X_test.shape)
print(f'Training Shape y:', y_train.shape, 'Testing Shape y:', y_test.shape)

# StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 4. Modeling
xgboost_model = xgb.XGBClassifier(n_estimators=100, max_depth=2)
xgboost_model.fit(X_train, y_train)

# 5. Optimization
#Placeholder for Optuna



# 6. Results interpretation (shap values, feature importance, partial dependence plot itp.)
y_pred = xgboost_model.predict(X_test)
log.debug('y_pred:', y_pred)

cm = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:',cm)

print(f'Accuracy:',accuracy_score(y_test, y_pred)* 100 ,'%')
print(classification_report(y_test, xgboost_model.predict(X_test)))


# 6.3. Explain the model's predictions using SHAP
explainer = shap.Explainer(xgboost_model, X_train)
shap_values = explainer(X_test)
shap_values.feature_names = feature_names
shap_values.target_names = target_names
print('SHAP values shape:', shap_values.shape)
#shap_values is an instance of Explanation object (https://shap.readthedocs.io/en/latest/generated/shap.Explanation.html#)
#It's an array of nested arrays. in our case it's:
#[91 - for each row a shap value is calculated]
#[*][6] - we have 6 features, so 6 columns
#[*][*][3] - we have 3 unique values of risk low, mid, high


#SHAP Summary Plot (global feature importance)
plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=feature_names, class_names=list(RiskLevelConversionMap.keys()))
plt.show()


# SHAP Summary Plot for High Risk feature importance
plt.figure()
shap.plots.bar(shap_values[:, :, 2])
plt.show()

# SHAP High Risk Beeswarm diagram for each feature
plt.figure()
shap.plots.beeswarm(shap_values[:, :, 2])
plt.show()

