import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

main_df = pd.read_csv("water_potability\\water_potability.csv")
df = main_df.copy()
print(df.describe())
