import pandas as pd
import os

print("Starte den finalen und kompletten Daten-Merge...")

# ---------------------------------------------------------
# 1. DAS TÄGLICHE RÜCKGRAT: Zielvariable (EXV9 ETF)
# ---------------------------------------------------------
df_master = pd.read_csv('data/EXV9_ETF.csv', header=[0, 1], index_col=0, parse_dates=True)
df_master = df_master['Close'].copy()
df_master.columns = ['EXV9_ETF_Kurs']
df_master.index.name = 'Datum'

# ---------------------------------------------------------
# 2. TÄGLICHE FINANZDATEN
# ---------------------------------------------------------
yfinance_dateien = [
    'DAX.csv', 'VIX.csv', 'BrentOil.csv', 'WTIOil.csv',
    'EUR_USD.csv', 'EUR_CHF.csv', 'EUR_GPB.csv', 'EUR_TRY.csv',
    'Lufthansa.csv', 'Ryanair.csv', 'EasyJet.csv', 'Air France-KLM.csv',
    'Booking.csv', 'Expedia.csv', 'Uber.csv', 'Airbnb.csv',
    'PrimeEnergyOil_Gas.csv', 'Marriott_Hotels.csv', 'Hilton_Hotels.csv', 'AWAY-ETF.csv'
]

print("Füge tägliche Finanzdaten hinzu...")
for datei in yfinance_dateien:
    pfad = f"data/{datei}"
    if os.path.exists(pfad):
        df_temp = pd.read_csv(pfad, header=[0, 1], index_col=0, parse_dates=True)
        df_temp = df_temp['Close'].copy()
        df_temp.columns = [datei.replace('.csv', '_Kurs')]
        df_temp.index.name = 'Datum'
        df_master = df_master.join(df_temp, how='left')

# ---------------------------------------------------------
# 3. TÄGLICHE MAKRO- UND NEWS-DATEN (INKL. SENTIMENT!)
# ---------------------------------------------------------
taegliche_dateien = {
    'gdelt_tourism_news.csv': ('datetime', ','),
    'google_news_sentiment.csv': ('date', ','),
    'google_trends.csv': ('date', ','),
    'wetter_openmeteo.csv': ('date', ','),
    'european_vacations_daily.csv': ('date', ','),
    'german_holidays_daily.csv': ('date', ',')
}

print("Füge tägliche Makro-, News- und Feriendaten hinzu...")
for datei, (datums_spalte, trenner) in taegliche_dateien.items():
    pfad = f"data/{datei}"
    if os.path.exists(pfad):
        try:
            # 1. Datei laden, OHNE das Datum direkt als Index zu blockieren
            df_temp = pd.read_csv(pfad, sep=trenner)

            # 2. Uhrzeit-Falle fixen: Alle Stunden/Minuten rigoros abschneiden
            df_temp[datums_spalte] = pd.to_datetime(df_temp[datums_spalte], errors='coerce').dt.normalize()

            # 3. GDELT-Fix: Falls es mehrere Einträge an einem Tag gibt, Durchschnitt bilden
            numerische_spalten = df_temp.select_dtypes(include='number').columns
            df_temp = df_temp.groupby(datums_spalte)[numerische_spalten].mean()
            df_temp.index.name = 'Datum'

            # 4. Wochenend-Falle fixen (Speziell für Google Trends)
            # Wenn Daten nur am Sonntag vorliegen, strecken wir sie BEVOR die Börsentage gefiltert werden
            if datei == 'google_trends.csv':
                df_temp = df_temp.resample('D').ffill()

            # 5. An die große Master-Tabelle ankleben
            df_master = df_master.join(df_temp, how='left')

        except Exception as e:
            print(f"  -> Warnung bei {datei}: {e}")

# ---------------------------------------------------------
# 4. WÖCHENTLICHE & MONATLICHE DATEN (MIT LÜCKENFÜLLER)
# ---------------------------------------------------------
langzeit_dateien = {
    'inflation.csv': ('date', ',', False),
    'ECDC_Europa_Inzidenz.csv': ('date', ',', True),
    'are_konsultationsinzidenz.csv': ('date', ',', True),
    'grippeweb.csv': ('date', ',', True)
}

print("Passe wöchentliche/monatliche Daten an das tägliche Raster an...")
for datei, (datums_spalte, trenner, is_weekly) in langzeit_dateien.items():
    pfad = f"data/{datei}"
    if os.path.exists(pfad):
        try:
            df_temp = pd.read_csv(pfad, sep=trenner)

            if is_weekly:
                df_temp[datums_spalte] = df_temp[datums_spalte].astype(str)
                df_temp[datums_spalte] = df_temp[datums_spalte].apply(
                    lambda x: x + '-1' if 'W' in x and len(x) == 8 else x
                )
                df_temp[datums_spalte] = pd.to_datetime(df_temp[datums_spalte], format='%G-W%V-%u', errors='coerce')
            else:
                df_temp[datums_spalte] = pd.to_datetime(df_temp[datums_spalte], errors='coerce')

            df_temp.set_index(datums_spalte, inplace=True)
            df_temp.index.name = 'Datum'

            df_master = df_master.join(df_temp, how='left')
        except Exception as e:
            print(f"  -> Warnung bei {datei}: {e}")

# ---------------------------------------------------------
# 5. LÜCKEN FÜLLEN (FORWARD-FILL)
# ---------------------------------------------------------
print("Strecke Wochen/Monatswerte auf die fehlenden Handelstage (Forward-Fill)...")
df_master.ffill(inplace=True)

# ---------------------------------------------------------
# 6. FINALES SPEICHERN
# ---------------------------------------------------------
ausgabe = 'finale_matrix.csv'
df_master.to_csv(ausgabe)

print(f"\n✅ LOGISCHER MERGE ERFOLGREICH! '{ausgabe}' wurde erstellt.")
print(f"Die Matrix hat {df_master.shape[0]} tägliche Zeilen und {df_master.shape[1]} Features.")