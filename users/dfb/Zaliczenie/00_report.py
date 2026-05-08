import pandas as pd
from ydata_profiling import ProfileReport

# 1. Wczytaj dane
df = pd.read_csv('data/data.csv', encoding='utf-8')

# 2. SZYBKIE CZYSZCZENIE TYLKO DLA RAPORTU (żeby Pearson widział liczby)
# Czyścimy przebieg
df['mileage'] = pd.to_numeric(df['mileage'].str.replace(' km', '').str.replace(' ', ''), errors='coerce')
# Czyścimy pojemność
df['engine_capacity'] = pd.to_numeric(df['engine_capacity'].str.replace(' cm3', '').str.replace(' ', ''), errors='coerce')
# Czyścimy rok
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# 3. Wybieramy tylko te kolumny, które chcemy na wykresie korelacji
cols_for_analysis = ['price_in_pln', 'mileage', 'engine_capacity', 'year', 'brand', 'gearbox', 'fuel_type']
df_subset = df[cols_for_analysis]

# 4. Generujemy raport na tym zestawie
report = ProfileReport(df_subset, title='Raport Korelacji', correlations={
    "pearson": {"calculate": True},
    "spearman": {"calculate": True}
})

report.to_file("data/auto_pro_report.html")