import streamlit as st
import pandas as pd
import pickle
import json
import os
from datetime import datetime
from urllib.parse import quote_plus

st.set_page_config(
    page_title="Rekomendacje film車w",
    page_icon="??",
    layout="wide"
)

# ========== ?CIE?KI ==========
HISTORY_FILE = 'app_files/history.json'

# ========== ?ADOWANIE MODELU I DANYCH ==========

@st.cache_resource
def load_model():
    with open('app_files/model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_movies():
    return pd.read_pickle('app_files/movies.pkl')

@st.cache_data
def load_artifacts():
    with open('app_files/top_directors.pkl', 'rb') as f:
        top_directors = pickle.load(f)
    with open('app_files/top_genres.pkl', 'rb') as f:
        top_genres = pickle.load(f)
    with open('app_files/features.pkl', 'rb') as f:
        features = pickle.load(f)
    return top_directors, top_genres, features

model = load_model()
df = load_movies()
top_directors, top_genres, features = load_artifacts()

# ========== HISTORIA WYSZUKIWA? ==========

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def save_to_history(entry):
    history = load_history()
    history.insert(0, entry)
    history = history[:20]
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# ========== POMOCNICZE FUNKCJE ==========

def youtube_search_url(title, year):
    query = quote_plus(f"{title} {int(year)} trailer")
    return f"https://www.youtube.com/results?search_query={query}"

def imdb_search_url(title):
    query = quote_plus(title)
    return f"https://www.imdb.com/find/?q={query}"

def poster_placeholder(title):
    clean_title = quote_plus(title[:40])
    return f"https://placehold.co/200x300/2a6099/white?text={clean_title}&font=roboto"

def get_poster_url(film):
    """Zwraca prawdziwy plakat z OMDb je?li jest, fallback na placeholder."""
    if 'Poster_URL' in film and pd.notna(film.get('Poster_URL')):
        return film['Poster_URL']
    return poster_placeholder(film['Series_Title'])

# ========== NAG?車WEK ==========

st.title("?? System rekomendacji film車w")
st.write(f"Baza **{len(df)} film車w**. Model: XGBoost z R2 = 0.54")

# ========== ZAK?ADKI ==========

tab_rekomendacje, tab_historia = st.tabs(["?? Rekomendacje", "?? Historia wyszukiwa里"])

# ========== SIDEBAR Z FILTRAMI ==========

st.sidebar.header("?? Twoje preferencje")
st.sidebar.write("Powiedz nam czego szukasz:")

wybrane_gatunki = st.sidebar.multiselect(
    "Ulubione gatunki:",
    options=top_genres,
    default=['Drama'],
    help="Wybierz jeden lub wi?cej gatunk車w"
)

# Dekada - shortcut albo w?asny zakres
min_rok = int(df['Released_Year'].min())
max_rok = int(df['Released_Year'].max())

dekada_preset = st.sidebar.selectbox(
    "Szybki wyb車r dekady:",
    options=['Dowolna', 'Lata 70.', 'Lata 80.', 'Lata 90.', 'Lata 2000.', 'Lata 2010.', 'Lata 2020.'],
    index=0
)

dekada_mapping = {
    'Lata 70.': (1970, 1979),
    'Lata 80.': (1980, 1989),
    'Lata 90.': (1990, 1999),
    'Lata 2000.': (2000, 2009),
    'Lata 2010.': (2010, 2019),
    'Lata 2020.': (2020, max_rok),
}

if dekada_preset == 'Dowolna':
    zakres_lat = st.sidebar.slider(
        "Lub ustaw w?asny zakres lat:",
        min_value=min_rok,
        max_value=max_rok,
        value=(2000, max_rok)
    )
else:
    zakres_lat = dekada_mapping[dekada_preset]
    st.sidebar.caption(f"Zakres: {zakres_lat[0]}每{zakres_lat[1]}")

# D?ugo?? filmu jako zakres
zakres_runtime = st.sidebar.slider(
    "D?ugo?? filmu (minuty):",
    min_value=60,
    max_value=240,
    value=(90, 180),
    step=15,
    help="Od jakiej do jakiej d?ugo?ci"
)

tylko_topowi_rezyserzy = st.sidebar.checkbox(
    "Tylko znani re?yserzy",
    value=False,
    help="Filmy wyre?yserowane przez top 20 re?yser車w w bazie"
)

min_glosow = st.sidebar.slider(
    "Minimalna liczba g?os車w IMDb:",
    min_value=0,
    max_value=500000,
    value=50000,
    step=10000,
    help="Wi?cej g?os車w = film bardziej znany"
)

min_metacritic = st.sidebar.slider(
    "Minimalny Metacritic Score:",
    min_value=0,
    max_value=100,
    value=0,
    step=5,
    help="0 = bez filtra. Metacritic to oceny krytyk車w"
)

liczba_wynikow = st.sidebar.slider(
    "Ile film車w pokaza?:",
    min_value=5,
    max_value=30,
    value=10
)

st.sidebar.markdown("---")
szukaj = st.sidebar.button("?? Znajd? filmy", type="primary", use_container_width=True)

# ========== ZAK?ADKA: REKOMENDACJE ==========

with tab_rekomendacje:
    if szukaj:
        if not wybrane_gatunki:
            st.warning("?? Wybierz przynajmniej jeden gatunek w panelu bocznym!")
            st.stop()

        # FILTROWANIE
        with st.spinner("Szukam film車w pasuj?cych do Twoich preferencji..."):
            filtered = df.copy()

            maska_gatunku = pd.Series([False] * len(filtered), index=filtered.index)
            for genre in wybrane_gatunki:
                maska_gatunku = maska_gatunku | (filtered[f'is_{genre}'] == 1)
            filtered = filtered[maska_gatunku]

            filtered = filtered[
                (filtered['Released_Year'] >= zakres_lat[0]) &
                (filtered['Released_Year'] <= zakres_lat[1])
            ]

            filtered = filtered[
                (filtered['Runtime'] >= zakres_runtime[0]) &
                (filtered['Runtime'] <= zakres_runtime[1])
            ]

            if tylko_topowi_rezyserzy:
                filtered = filtered[filtered['is_top_director'] == 1]

            filtered = filtered[filtered['No_of_Votes'] >= min_glosow]

            if min_metacritic > 0:
                filtered = filtered[filtered['Meta_score'] >= min_metacritic]

            if len(filtered) == 0:
                st.error("? ?aden film nie pasuje do Twoich filtr車w. Rozlu?nij kryteria.")
                st.stop()

        # PREDYKCJA MODELEM
        with st.spinner(f"Znaleziono {len(filtered)} pasuj?cych film車w. Model ocenia..."):
            X_filtered = filtered[features]
            filtered = filtered.copy()
            filtered['Przewidywana_ocena'] = model.predict(X_filtered).round(2)

        top_filmy = filtered.sort_values('Przewidywana_ocena', ascending=False).head(liczba_wynikow)

        # ZAPIS DO HISTORII
        entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'gatunki': wybrane_gatunki,
            'lata': f"{zakres_lat[0]}-{zakres_lat[1]}",
            'dlugosc': f"{zakres_runtime[0]}-{zakres_runtime[1]} min",
            'min_glosow': min_glosow,
            'min_metacritic': min_metacritic,
            'topowi_rezyserzy': tylko_topowi_rezyserzy,
            'liczba_wynikow': len(top_filmy),
            'top_3': top_filmy['Series_Title'].head(3).tolist()
        }
        save_to_history(entry)

        # WY?WIETLENIE
        st.success(f"? Znalaz?em {len(top_filmy)} film車w dla Ciebie!")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pasuj?ce filmy", len(filtered))
        col2.metric("?rednia IMDb w wynikach", f"{top_filmy['IMDB_Rating'].mean():.2f}")
        col3.metric("Najwy?sza predykcja", f"{top_filmy['Przewidywana_ocena'].max():.2f}")
        col4.metric("?redni Metacritic", f"{top_filmy['Meta_score'].mean():.0f}")

        st.markdown("---")

        # Lista film車w z plakatami
        for idx, (_, film) in enumerate(top_filmy.iterrows(), start=1):
            with st.container():
                kol_plakat, kol_info, kol_oceny = st.columns([1.5, 4, 2])

                with kol_plakat:
                    st.image(get_poster_url(film), width=180)

                with kol_info:
                    st.markdown(f"### #{idx}. {film['Series_Title']} ({int(film['Released_Year'])})")
                    st.write(f"?? **Gatunek:** {film['Genre']}")
                    st.write(f"?? **Re?yser:** {film['Director']}")
                    st.write(f"?? **D?ugo??:** {int(film['Runtime'])} min  |  ?? **G?osy:** {int(film['No_of_Votes']):,}")
                    if pd.notna(film['Meta_score']):
                        st.write(f"?? **Metacritic:** {int(film['Meta_score'])}/100")

                    # Plot z OMDb je?li jest
                    if 'Plot' in film and pd.notna(film.get('Plot')):
                        st.caption(f"?? {film['Plot']}")

                    # Linki zewn?trzne
                    yt_url = youtube_search_url(film['Series_Title'], film['Released_Year'])
                    imdb_url = imdb_search_url(film['Series_Title'])
                    st.markdown(
                        f"??? [Zobacz trailer na YouTube]({yt_url}) &nbsp;?&nbsp; "
                        f"?? [Znajd? na IMDb]({imdb_url})"
                    )

                with kol_oceny:
                    st.metric("Ocena IMDb", film['IMDB_Rating'])
                    st.metric("Predykcja modelu", film['Przewidywana_ocena'])

                st.markdown("---")

    else:
        st.info("?? Ustaw preferencje w panelu bocznym i kliknij **Znajd? filmy**")

        with st.expander("Zobacz przyk?adowe filmy z bazy"):
            st.dataframe(df[['Series_Title', 'Released_Year', 'Genre', 'IMDB_Rating', 'Director']].sample(10))

# ========== ZAK?ADKA: HISTORIA ==========

with tab_historia:
    st.header("?? Historia ostatnich wyszukiwa里")
    st.caption("Twoje ostatnie 20 wyszukiwa里 (zapisywane automatycznie)")

    history = load_history()

    if not history:
        st.info("Nie ma jeszcze ?adnej historii. Przeprowad? pierwsze wyszukiwanie w zak?adce Rekomendacje.")
    else:
        col_a, col_b = st.columns([5, 1])
        with col_b:
            if st.button("??? Wyczy?? histori?", use_container_width=True):
                if os.path.exists(HISTORY_FILE):
                    os.remove(HISTORY_FILE)
                st.rerun()

        st.markdown("---")

        for i, entry in enumerate(history, start=1):
            with st.expander(f"#{i} 〞 {entry['timestamp']} 〞 {', '.join(entry['gatunki'])}", expanded=(i == 1)):
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Gatunki:** {', '.join(entry['gatunki'])}")
                    st.write(f"**Lata:** {entry['lata']}")
                    st.write(f"**D?ugo??:** {entry['dlugosc']}")
                with c2:
                    st.write(f"**Min. g?os車w:** {entry['min_glosow']:,}")
                    st.write(f"**Min. Metacritic:** {entry['min_metacritic']}")
                    st.write(f"**Tylko top re?yserzy:** {'Tak' if entry['topowi_rezyserzy'] else 'Nie'}")

                st.write(f"**Top 3 filmy z tego wyszukiwania:**")
                for j, title in enumerate(entry['top_3'], start=1):
                    st.write(f"  {j}. {title}")
