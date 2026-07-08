import pandas as pd
import os
from pathlib import Path

# Sicherstellen, dass die Pfade immer relativ zum Skript aufgelöst werden
SCRIPT_DIR = Path(__file__).parent
PROJEKT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJEKT_ROOT / 'data'
DATA_PATH = DATA_DIR / 'EXV9_ETF.csv'

print("Starte den finalen und kompletten Daten-Merge...")

# ---------------------------------------------------------
# 1. DAS TÄGLICHE RÜCKGRAT: Zielvariable (EXV9 ETF)
# ---------------------------------------------------------
df_master = pd.read_csv(DATA_PATH, header=[0, 1], index_col=0, parse_dates=True)
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
    pfad = DATA_DIR / datei
    if pfad.exists():
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
    'european_vacations_daily.csv': ('date', ','),
    'german_holidays_daily.csv': ('date', ',')
}

print("Füge tägliche Makro-, News- und Feriendaten hinzu...")
for datei, (datums_spalte, trenner) in taegliche_dateien.items():
    pfad = DATA_DIR / datei
    if pfad.exists():
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
pfad = DATA_DIR / "inflation.csv"

if pfad.exists():
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

        monats_spalten = [
            spalte for spalte in df_inflation.columns
            if str(spalte).startswith("20")
        ]

        df_inflation_long = df_inflation.melt(
            id_vars=["coicop"],
            value_vars=monats_spalten,
            var_name="Monat",
            value_name="Wert"
        )

        df_inflation_long["Wert"] = pd.to_numeric(
            df_inflation_long["Wert"],
            errors="coerce"
        )

        df_inflation_long["Datum"] = pd.to_datetime(
            df_inflation_long["Monat"],
            format="%Y-%m",
            errors="coerce"
        )

        df_inflation_long["Variable"] = df_inflation_long["coicop"].map(
            relevante_coicop
        )

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

        df_inflation_daily = (
            df_inflation_monthly
            .reindex(df_master.index)
            .ffill()
            .bfill()
        )

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
pfad = DATA_DIR / "ECDC_Europa_Inzidenz.csv"

if pfad.exists():
    try:
        df_ecdc = pd.read_csv(pfad, sep=",")

        df_ecdc = df_ecdc[
            df_ecdc["age"].astype(str).str.lower() == "total"
        ].copy()

        df_ecdc["value"] = pd.to_numeric(
            df_ecdc["value"],
            errors="coerce"
        )

        df_ecdc_weekly = (
            df_ecdc
            .groupby("yearweek", as_index=False)["value"]
            .mean()
            .rename(columns={"value": "ECDC_Europa_Inzidenz"})
        )

        df_ecdc_weekly["ECDC_Europa_Inzidenz"] = (
            df_ecdc_weekly["ECDC_Europa_Inzidenz"]
            .round()
            .astype(int)
        )

        df_ecdc_weekly["Datum"] = pd.to_datetime(
            df_ecdc_weekly["yearweek"] + "-1",
            format="%G-W%V-%u",
            errors="coerce"
        )

        df_ecdc_weekly.dropna(subset=["Datum"], inplace=True)
        df_ecdc_weekly.set_index("Datum", inplace=True)
        df_ecdc_weekly = df_ecdc_weekly[["ECDC_Europa_Inzidenz"]]

        df_ecdc_daily = (
            df_ecdc_weekly
            .reindex(df_master.index)
            .ffill()
            .bfill()
        )

        df_ecdc_daily["ECDC_Europa_Inzidenz"] = (
            df_ecdc_daily["ECDC_Europa_Inzidenz"]
            .round()
            .astype(int)
        )

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
# WETTERDATEN (STÄDTE -> AGGREGIERTE WETTER-FEATURES)
# ---------------------------------------------------------
print("Füge aggregierte Wetterdaten hinzu...")
pfad = DATA_DIR / "wetter_openmeteo.csv"

if pfad.exists():
    try:
        df_weather = pd.read_csv(pfad, sep=",")

        df_weather["Datum"] = pd.to_datetime(
            df_weather["time"],
            errors="coerce"
        ).dt.normalize()

        central_cities = ["Berlin", "Paris", "Amsterdam", "Prag", "Wien"]
        holiday_cities = ["Rom", "Madrid", "Barcelona", "Palma", "Antalya"]

        df_weather = df_weather[
            df_weather["city"].isin(central_cities + holiday_cities)
        ].copy()

        df_weather["weather_group"] = df_weather["city"].apply(
            lambda x: "CentralEurope" if x in central_cities else "HolidayDestination"
        )

        df_weather["sunshine_hours"] = df_weather["sunshine_duration"] / 3600

        df_weather_grouped = (
            df_weather
            .groupby(["Datum", "weather_group"])
            .agg({
                "temperature_2m_mean": "mean",
                "sunshine_hours": "mean",
                "precipitation_sum": "mean"
            })
            .reset_index()
        )

        df_weather_wide = df_weather_grouped.pivot(
            index="Datum",
            columns="weather_group"
        )

        df_weather_wide.columns = [
            f"{group}_{metric}"
            for metric, group
            in df_weather_wide.columns
        ]

        df_weather_wide.rename(columns={
            "CentralEurope_temperature_2m_mean": "CentralEurope_Temp",
            "CentralEurope_sunshine_hours": "CentralEurope_Sunshine",
            "CentralEurope_precipitation_sum": "CentralEurope_Precipitation",
            "HolidayDestination_temperature_2m_mean": "HolidayDestination_Temp",
            "HolidayDestination_sunshine_hours": "HolidayDestination_Sunshine",
            "HolidayDestination_precipitation_sum": "HolidayDestination_Precipitation"
        }, inplace=True)

        df_weather_wide = df_weather_wide.round({
            "CentralEurope_Temp": 1,
            "HolidayDestination_Temp": 1,
            "CentralEurope_Sunshine": 1,
            "HolidayDestination_Sunshine": 1,
            "CentralEurope_Precipitation": 1,
            "HolidayDestination_Precipitation": 1
        })

        df_master = df_master.join(df_weather_wide, how='left')

        print(
            f"  -> Wetterdaten erfolgreich integriert "
            f"({len(df_weather_wide)} Tageswerte)"
        )

    except Exception as e:
        print(f"  -> Warnung bei wetter_openmeteo.csv: {e}")
else:
    print("  -> wetter_openmeteo.csv nicht gefunden")

# ---------------------------------------------------------
# 5. LÜCKEN FÜLLEN (FORWARD-FILL)
# ---------------------------------------------------------
print("Strecke Wochen/Monatswerte auf die fehlenden Handelstage (Forward-Fill)...")
df_master.ffill(inplace=True)

# ---------------------------------------------------------
# 6. FINALES SPEICHERN
# ---------------------------------------------------------
# Speichert die Datei absolut sicher im Ordner 'data-prep-pre-split' ab
ausgabe = SCRIPT_DIR / 'raw-data-matrix.csv'
df_master.to_csv(ausgabe)
print(f"Datei erfolgreich gespeichert unter: {ausgabe}")