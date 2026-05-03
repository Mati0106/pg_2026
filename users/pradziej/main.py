import logging as log
from src.DataLoader import DataLoader
from src.settings import settings
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
import pandas as pd

from sklearn.preprocessing import StandardScaler


log.basicConfig(level=log.DEBUG)

# 1. Read the data
dataset_name = settings['dataset']
data_loader = DataLoader()
df = data_loader.load_data(dataset_name, force_download=False)
log.info(df.describe())
log.info(print(df.head()))

# 2. Data analysis + preprocessing
#Data understanding
data_understanding_explanation = """
Data has been collected from different hospitals, community clinics, maternal health cares through the IoT based risk monitoring system.

Columns:
 - Age - int,
 - SystolicBP - Upper value of Blood Pressure in mmHg
 - DiastolicBP - Lower value of Blood Pressure in mmHg
 - BS - Blood glucose levels is in terms of a molar concentration, mmol/L
 - BodyTemp - in F,
 - HeartRate - A normal resting heart rate in beats per minute.
 - RiskLevel - Predicted Risk Intensity Level during pregnancy -> THIS WILL BE OUR TARGET


 MAIN QUESTION:
  - Based on the simple mearusrements, that each future mum can make at home, provide a risk rank (high, medium, low)
"""
log.info(data_understanding_explanation)

log.debug(f"Columns, {df.columns}")
log.debug(f'RiskLevel:\n{df['RiskLevel'].unique()}')


df.drop_duplicates()

# 3.1 BodyTemp is in Farenheit - convert it to Celsius
def to_celcius(temp_f):
    return round((temp_f - 32) / 1.8, 1)

df['BodyTempC'] = df['BodyTemp'].apply(to_celcius)
log.debug(f'BodyTempC:\n\t{df['BodyTempC'].describe()}')
df = df.drop(columns=['BodyTemp'])


# Convert a subset of columns to categorical
RiskLevel = {
    'low risk': 1, 
    'mid risk': 2, 
    'high risk': 3
}
df['RiskLevel'] = df['RiskLevel'].map(RiskLevel).astype(float)


# 3. Feature engineering
def get_age_group(age):
    if age < 18:
        return 'Under 18'
    elif age >= 18 and age < 25:
        return '18-24'
    elif age >= 25 and age < 35:
        return '25-34'
    elif age >= 35 and age < 45:
        return '35-44'
    elif age >= 45 and age < 55:
        return '45-54'
    elif age >= 55 and age < 65:
        return '55-64'
    else:
        return '65+'
#df['AgeGroup'] = df['Age'].apply(get_age_group)
#log.debug(f'AgeGroup:\n{df['AgeGroup'].describe()}')
#df_filtered = df.drop(columns=['Age'])



target_names = ['RiskLevel']
feature_names = [c for c in df.columns if c not in target_names]
log.info(f'Feature names: {feature_names}, Target names: {target_names}')


X = df[feature_names]
y = df[target_names]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Convert data into DMatrix format for XGBoost - seams it's not needed
#dtrain = xgb.DMatrix(X_train, label=y_train, enable_cagegorical=True)
#dtest = xgb.DMatrix(X_test, label=y_test, enable_cagegorical=True)


#Shape
print(f'Training Shape x:',X_train.shape)
print(f'Testing Shape x:',X_test.shape)
print('*****___________*****___________*****')
print(f'Training Shape y:',X.shape)
print(f'Testing Shape y:',y.shape)



#StandardScaler

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test= ss.transform(X_test)



# 4. Modeling
xgboost_model = xgb.XGBClassifier(enable_cagegorical=True)
xgboost_model.fit(X_train, y_train)






# 5. Optimization









# 6. Results interpretation (Wartości shapley'a, feature importance, partial dependence plot itp.)
# 6.3. Explain the model's predictions using SHAP
explainer = shap.Explainer(xgboost_model, X_train)
shap_values = explainer(X_test)

# 6.4. SHAP Summary Plot (global feature importance)
plt.figure()  # Create a new figure
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
plt.show()  # Display the plot

# 6.5. SHAP Dependence Plot (feature vs. SHAP value)
shap.dependence_plot('MedInc', shap_values.values, X_test, feature_names=feature_names)
plt.show()  # Display the plot

# 6.6. SHAP Force Plot (local explanation of a single prediction)
# Create a new figure
shap.force_plot(explainer.expected_value, shap_values[0].values, X_test[0], feature_names=feature_names, matplotlib=True)
plt.show()  # Display the plot

# 6.7. SHAP Waterfall Plot (breakdown of individual prediction)
plt.figure()  # Create a new figure
shap.plots.waterfall(shap_values[0])
plt.show()  # Display the plot



