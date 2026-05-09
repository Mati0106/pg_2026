# Zbiór danych do przewidywania niewydolności serca

## 11 cech klinicznych do przewidywania występowania chorób serca

---

# Informacje o zbiorze danych

## Podobne zbiory danych
- Dataset wirusowego zapalenia wątroby typu C  
- Dataset przewidywania poziomu tkanki tłuszczowej  
- Dataset przewidywania marskości wątroby  
- Dataset przewidywania udaru  
- Dataset klasyfikacji gwiazd SDSS17  
- Dataset przewidywania prędkości wiatru  
- Dataset jakości hiszpańskiego wina  

---

# Kontekst

Choroby układu sercowo-naczyniowego (CVD — *Cardiovascular Diseases*) są główną przyczyną śmierci na świecie. Każdego roku powodują około 17,9 miliona zgonów, co stanowi 31% wszystkich zgonów globalnie.

Cztery na pięć zgonów związanych z CVD wynikają z:
- zawałów serca,
- udarów mózgu,

a jedna trzecia tych zgonów występuje przedwcześnie u osób poniżej 70. roku życia.

Niewydolność serca jest częstym skutkiem chorób sercowo-naczyniowych, a ten zbiór danych zawiera 11 cech, które mogą zostać wykorzystane do przewidywania możliwej choroby serca.

Osoby cierpiące na choroby sercowo-naczyniowe lub znajdujące się w grupie wysokiego ryzyka (z powodu występowania jednego lub więcej czynników ryzyka, takich jak:
- nadciśnienie,
- cukrzyca,
- hiperlipidemia,
- wcześniej zdiagnozowana choroba serca)

wymagają wczesnego wykrywania i odpowiedniego leczenia — w czym model uczenia maszynowego może być bardzo pomocny.

---

# Informacje o atrybutach

### Age → Wiek
Wiek pacjenta [lata]

---

### Sex → Płeć
[M: mężczyzna, F: kobieta]

---

### ChestPainType → Typ bólu w klatce piersiowej
- TA: typowa dusznica bolesna  
- ATA: atypowa dusznica bolesna  
- NAP: ból niedławicowy  
- ASY: bezobjawowy  

---

### RestingBP → Ciśnienie krwi w spoczynku
[mm Hg]

---

### Cholesterol → Poziom cholesterolu w surowicy
[mg/dl]

---

### FastingBS → Poziom cukru we krwi na czczo
- 1: jeśli poziom > 120 mg/dl  
- 0: w przeciwnym razie  

---

### RestingECG → Wyniki EKG w spoczynku
- Normal: wynik prawidłowy  
- ST: nieprawidłowości odcinka ST-T  
  (odwrócenie załamka T i/lub podwyższenie albo obniżenie odcinka ST > 0,05 mV)  
- LVH: prawdopodobny lub pewny przerost lewej komory serca według kryteriów Estesa  

---

### MaxHR → Maksymalne osiągnięte tętno
[wartość liczbowa od 60 do 202]

---

### ExerciseAngina → Dusznica wywołana wysiłkiem
- Y: tak  
- N: nie  

---

### Oldpeak → Oldpeak = ST
[wartość liczbowa określająca obniżenie odcinka ST]

---

### ST_Slope → Nachylenie odcinka ST podczas maksymalnego wysiłku
- Up: rosnące  
- Flat: płaskie  
- Down: opadające  

---

### HeartDisease → Klasa wyjściowa
- 1: choroba serca  
- 0: stan prawidłowy  

---

# Źródło

Ten zbiór danych został utworzony poprzez połączenie kilku różnych datasetów, które wcześniej były dostępne oddzielnie, ale nigdy nie zostały scalone.

W zbiorze połączono:
- 5 datasetów dotyczących chorób serca,
- 11 wspólnych cech,

co czyni go największym dostępnym zbiorem danych o chorobach serca wykorzystywanym do celów badawczych.

## Zbiory wykorzystane do stworzenia datasetu:

- Cleveland — 303 obserwacje  
- Hungarian — 294 obserwacje  
- Switzerland — 123 obserwacje  
- Long Beach VA — 200 obserwacji  
- Stalog (Heart) Dataset — 270 obserwacji  
