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
# INFLATION EA20 (MONATSDATEN -> TAGESDATEN)
# ---------------------------------------------------------

print("Passe Inflationsdaten an das tägliche Raster an...")

pfad = "data/inflation.csv"

if os.path.exists(pfad):
    try:
        df_inflation = pd.read_csv(pfad, sep=",")

        # Nur jährliche Inflationsrate und relevante COICOP-Kategorien
        relevante_coicop = {
            "CP00": "Inflation_Gesamt",
            "CP07": "Inflation_Transport",
            "CP11": "Inflation_Restaurants_Hotels"
        }

        df_inflation = df_inflation[
            (df_inflation["unit"] == "I15") &
            (df_inflation["coicop"].isin(relevante_coicop.keys()))
        ].copy()

        # Monatsspalten erkennen, z.B. 2023-01, 2023-02, ...
        monats_spalten = [
            spalte for spalte in df_inflation.columns
            if str(spalte).startswith("20")
        ]

        # Von breitem Format in langes Format umwandeln
        df_inflation_long = df_inflation.melt(
            id_vars=["coicop"],
            value_vars=monats_spalten,
            var_name="Monat",
            value_name="Wert"
        )

        # Monatswerte numerisch machen
        df_inflation_long["Wert"] = pd.to_numeric(
            df_inflation_long["Wert"],
            errors="coerce"
        )

        # Monat in Datum umwandeln
        df_inflation_long["Datum"] = pd.to_datetime(
            df_inflation_long["Monat"],
            format="%Y-%m",
            errors="coerce"
        )

        # Spaltennamen lesbarer machen
        df_inflation_long["Variable"] = df_inflation_long["coicop"].map(
            relevante_coicop
        )

        # Zurück in breites Format: eine Spalte je Inflationskategorie
        df_inflation_monthly = (
            df_inflation_long
            .pivot_table(
                index="Datum",
                columns="Variable",
                values="Wert",
                aggfunc="mean"
            )
            .sort_index()
        )

        df_inflation_monthly.columns.name = None

        # Auf tägliches Raster von df_master erweitern
        df_inflation_daily = (
            df_inflation_monthly
            .reindex(df_master.index)
            .ffill()
            .bfill()
        )

        # In Hauptmatrix integrieren
        df_master = df_master.join(
            df_inflation_daily,
            how="left"
        )

        print(
            f"  -> Inflationsdaten erfolgreich integriert "
            f"({len(df_inflation_monthly)} Monatswerte)"
        )

    except Exception as e:
        print(f"  -> Warnung bei inflation.csv: {e}")

else:
    print("  -> inflation.csv nicht gefunden")

# ---------------------------------------------------------
# ECDC EUROPA INZIDENZ (WOCHENDATEN -> TAGESDATEN)
# ---------------------------------------------------------

print("Passe ECDC-Wochendaten an das tägliche Raster an...")


pfad = "data/ECDC_Europa_Inzidenz.csv"

if os.path.exists(pfad):
    try:
        # CSV laden
        df_ecdc = pd.read_csv(pfad, sep=",")

        # Nur Gesamtbevölkerung verwenden
        df_ecdc = df_ecdc[
            df_ecdc["age"].astype(str).str.lower() == "total"
        ].copy()

        # Inzidenz numerisch machen
        df_ecdc["value"] = pd.to_numeric(
            df_ecdc["value"],
            errors="coerce"
        )

        # Durchschnitt aller Länder pro Woche bilden
        df_ecdc_weekly = (
            df_ecdc
            .groupby("yearweek", as_index=False)["value"]
            .mean()
            .rename(columns={"value": "ECDC_Europa_Inzidenz"})
        )

        # Auf ganze Zahlen runden
        df_ecdc_weekly["ECDC_Europa_Inzidenz"] = (
            df_ecdc_weekly["ECDC_Europa_Inzidenz"]
            .round()
            .astype(int)
        )

        # ISO-Woche -> Datum (Montag der jeweiligen Woche)
        df_ecdc_weekly["Datum"] = pd.to_datetime(
            df_ecdc_weekly["yearweek"] + "-1",
            format="%G-W%V-%u",
            errors="coerce"
        )

        # Ungültige Datumswerte entfernen
        df_ecdc_weekly.dropna(subset=["Datum"], inplace=True)

        # Datum als Index setzen
        df_ecdc_weekly.set_index("Datum", inplace=True)

        # Nur die fertige Zielspalte behalten
        df_ecdc_weekly = df_ecdc_weekly[
            ["ECDC_Europa_Inzidenz"]
        ]

        # Auf tägliches Raster von df_master erweitern
        df_ecdc_daily = (
            df_ecdc_weekly
            .reindex(df_master.index)
            .ffill()
            .bfill()
        )

        # Erneut sicherstellen, dass Integer verwendet werden
        df_ecdc_daily["ECDC_Europa_Inzidenz"] = (
            df_ecdc_daily["ECDC_Europa_Inzidenz"]
            .round()
            .astype(int)
        )

        # In Hauptmatrix integrieren
        df_master = df_master.join(
            df_ecdc_daily,
            how="left"
        )

        print(
            f"  -> ECDC-Daten erfolgreich integriert "
            f"({len(df_ecdc_weekly)} Wochenwerte)"
        )

    except Exception as e:
        print(f"  -> Warnung bei ECDC_Europa_Inzidenz.csv: {e}")

else:
    print("  -> ECDC_Europa_Inzidenz.csv nicht gefunden")

# ---------------------------------------------------------
# 5. LÜCKEN FÜLLEN (FORWARD-FILL)
# ---------------------------------------------------------
print("Strecke Wochen/Monatswerte auf die fehlenden Handelstage (Forward-Fill)...")
df_master.ffill(inplace=True)

# ---------------------------------------------------------
# 6. FINALES SPEICHERN
# ---------------------------------------------------------
ausgabe = './data-prep-pre-split/raw-data-matrix.csv'
df_master.to_csv(ausgabe)
