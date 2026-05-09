from src.preprocessing import preprocess
from src.scripts import eda
from src.modelling import train
from src.scripts import evaluate
from src.scripts import forecast

# Step 1 - load data
print("Step 1: Wczytanie danych")
df_raw = preprocess.load_data()

# Step 2 - exploratory data analysis (BEV-filtered + raw for charging stations)
print("\nStep 2: Analiza danych")
df_bev = preprocess.filter_bev(df_raw)
eda.run(df_bev, df_raw)

# Step 3 - preprocessing and feature engineering (including charging stations)
print("\nStep 3: Preprocessing i feature engineering")
preprocess.run()

# Step 4 & 5 - model training and hyperparameter optimization
print("\nStep 4 & 5: Model i optymalizacja hiperparametrow")
train.run()

# Step 6 - results interpretation and visualizations
print("\nStep 6: Interpretacja wynikow i wizualizacje")
evaluate.run()

# Step 7 - Poland 2030 forecast (BEV adoption + charging stations projection)
print("\nStep 7: Prognoza dla Polski do 2030")
forecast.run()

print("\nGotowe!")
