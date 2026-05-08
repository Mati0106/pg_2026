import pandas as pd
import numpy as np
import json

# 1. Wczytanie danych
df = pd.read_csv('data/data.csv', encoding='utf-8')

# 2. Czyszczenie Wojewódźtw (Tylko 16 polskich województw)
polish_voivodeships = [
    'Dolnośląskie', 'Kujawsko-pomorskie', 'Lubelskie', 'Lubuskie', 'Łódzkie',
    'Małopolskie', 'Mazowieckie', 'Opolskie', 'Podkarpackie', 'Podlaskie',
    'Pomorskie', 'Śląskie', 'Świętokrzyskie', 'Warmińsko-mazurskie',
    'Wielkopolskie', 'Zachodniopomorskie'
]
df = df[df['voivodeship'].isin(polish_voivodeships)]

# 3. Czyszczenie danych numerycznych
df['mileage'] = pd.to_numeric(df['mileage'].str.replace(' km', '').str.replace(' ', ''), errors='coerce')
df['engine_capacity'] = pd.to_numeric(df['engine_capacity'].str.replace(' cm3', '').str.replace(' ', ''), errors='coerce')
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['price_in_pln', 'mileage', 'year', 'engine_capacity'])

# 4. Inteligentne wyciąganie nazwy modelu (np. z "Audi A4 Avant" robi "A4")
def clean_model_name(row):
    brand = str(row['brand']).lower()
    model_str = str(row['model']).lower()
    if model_str.startswith(brand):
        model_str = model_str[len(brand):].strip()
    parts = model_str.split()
    return parts[0].capitalize() if parts else 'Inny'

df['model_refined'] = df.apply(clean_model_name, axis=1)

# Zostawiamy modele, które mają min. 15 ogłoszeń
counts = df['model_refined'].value_counts()
popular_models = counts[counts >= 15].index
df['model_final'] = df['model_refined'].apply(lambda x: x if x in popular_models else 'Inny')

# 5. Przygotowanie cech do uczenia
df['age'] = 2026 - df['year']
cols_to_keep = ['brand', 'model_final', 'gearbox', 'fuel_type', 'voivodeship', 'mileage', 'engine_capacity', 'age', 'price_in_pln']
df_final = df[cols_to_keep]

# 6. Kodowanie One-Hot (zamiana tekstu na kolumny 0/1)
df_ml = pd.get_dummies(df_final, columns=['brand', 'model_final', 'gearbox', 'fuel_type', 'voivodeship'], drop_first=True)

# Zapis do CSV
df_ml.to_csv('data/cleaned_data.csv', index=False)

# 7. Zapisywanie metadanych do JSON do obsługi Streamlit
metadata_json = {
    'brands': sorted(df_final['brand'].unique().tolist()),
    'brand_models': df_final[['brand', 'model_final']].drop_duplicates().values.tolist(),
    'voivodeships': sorted(polish_voivodeships),
    'fuels': sorted(df_final['fuel_type'].unique().tolist()),
    'gearboxes': sorted(df_final['gearbox'].unique().tolist())
}

with open('data/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata_json, f, ensure_ascii=False, indent=4)

print(f"Dane gotowe! Liczba rekordów: {len(df_ml)}")