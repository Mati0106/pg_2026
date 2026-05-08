import streamlit as st
import pandas as pd
import xgboost as xgb
import json
import numpy as np

# KONFIGURACJA STRONY
st.set_page_config(page_title="AI Kalkulator Wyceny Aut", page_icon="🚗")


# 1. FUNKCJA ŁADOWANIA ZASOBÓW (Cache sprawia, że aplikacja działa błyskawicznie)
@st.cache_resource
def load_resources():
    # Wczytujemy model
    model = xgb.XGBRegressor()
    model.load_model('data/car_model.json')

    # Wczytujemy metadane (listy marek, modeli itp.)
    with open('data/metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Wczytujemy nagłówki kolumn, żeby wiedzieć jak przygotować dane dla modelu
    # Używamy tylko pierwszego wiersza, żeby nie marnować pamięci RAM
    df_sample = pd.read_csv('data/cleaned_data.csv', nrows=1)
    ml_columns = df_sample.drop('price_in_pln', axis=1).columns

    return model, metadata, ml_columns


# Ładujemy wszystko na start
try:
    model, metadata, ml_columns = load_resources()
except FileNotFoundError:
    st.error("Błąd: Nie znaleziono plików modelu lub danych w folderze /data. Uruchom najpierw skrypty 1 i 2!")
    st.stop()

# 2. INTERFEJS UŻYTKOWNIKA
st.title("🚗 System Inteligentnej Wyceny Samochodów")
st.markdown("Wprowadź parametry pojazdu, aby otrzymać szacunkową wycenę rynkową wygenerowaną przez AI.")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Marka pojazdu", metadata['brands'])

    # Dynamiczna lista modeli - pokazuje tylko te, które należą do wybranej marki
    available_models = [m[1] for m in metadata['brand_models'] if m[0] == brand]
    model_car = st.selectbox("Model", sorted(available_models))

    year = st.number_input("Rok produkcji", min_value=1990, max_value=2025, value=2015)
    age = 2026 - year

with col2:
    engine = st.number_input("Pojemność silnika (cm3)", min_value=500, max_value=6000, value=1968, step=100)
    mileage = st.number_input("Przebieg (km)", min_value=0, max_value=500000, value=150000, step=5000)
    fuel = st.selectbox("Rodzaj paliwa", metadata['fuels'])
    gearbox = st.selectbox("Skrzynia biegów", metadata['gearboxes'])

voivodeship = st.selectbox("Województwo sprzedaży", metadata['voivodeships'])

# 3. PRZYGOTOWANIE DANYCH DO PREDYKCJI
if st.button("WYCENA AUTA", type="primary", use_container_width=True):
    # Tworzymy pusty wiersz z samymi zerami o strukturze takiej jak w treningu
    input_df = pd.DataFrame(0, index=[0], columns=ml_columns)

    # Wypełniamy wartości numeryczne
    input_df['mileage'] = mileage
    input_df['engine_capacity'] = engine
    input_df['age'] = age

    # Wypełniamy kolumny One-Hot (ustawiamy 1 tam, gdzie użytkownik dokonał wyboru)
    # Pamiętamy o drop_first=True - jeśli kolumny nie ma w input_df, to znaczy że była tą "pierwszą" i jest OK
    if f'brand_{brand}' in input_df.columns:
        input_df[f'brand_{brand}'] = 1
    if f'model_final_{model_car}' in input_df.columns:
        input_df[f'model_final_{model_car}'] = 1
    if f'fuel_type_{fuel}' in input_df.columns:
        input_df[f'fuel_type_{fuel}'] = 1
    if f'gearbox_{gearbox}' in input_df.columns:
        input_df[f'gearbox_{gearbox}'] = 1
    if f'voivodeship_{voivodeship}' in input_df.columns:
        input_df[f'voivodeship_{voivodeship}'] = 1

    # 4. PREDYKCJA
    prediction = model.predict(input_df)[0]

    # Zabezpieczenie przed nierealistycznymi wynikami (np. ujemna cena)
    final_price = max(prediction, 1000)

    st.divider()
    st.subheader(f"Przewidywana wartość rynkowa:")
    st.header(f"{final_price:,.2f} PLN".replace(",", " "))

    st.info(
        "ℹ️ Powyższa wycena jest generowana przez model uczenia maszynowego (XGBoost) na podstawie danych historycznych.")