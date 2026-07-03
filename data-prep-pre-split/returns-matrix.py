import pandas as pd
from pathlib import Path

# Sicherstellen, dass die Pfade immer relativ zum Skript aufgelöst werden
SCRIPT_DIR = Path(__file__).parent

# Pfad zur Eingabe-Datei (liegt im selben Ordner)
INPUT_PATH = SCRIPT_DIR / "raw-data-matrix.csv"

print("Lade die Rohdaten-Matrix...")
df = pd.read_csv(INPUT_PATH)


def create_return_features(df, columns):
    """
    Erstellt Return-Features für die angegebenen Spalten.
    """
    df_returns = pd.DataFrame()
    df_returns["Datum"] = df["Datum"]

    for col in columns:
        # Falls eine Spalte in den Daten fehlt, wird sie übersprungen,
        # damit das Skript nicht abstürzt
        if col in df.columns:
            df_returns[f"{col}_return_1d"] = df[col].pct_change(1) * 100
            df_returns[f"{col}_return_5d"] = df[col].pct_change(5) * 100
        else:
            print(f"  -> Warnung: Spalte '{col}' nicht in raw-data-matrix.csv gefunden!")

    return df_returns


return_columns = [
    "EXV9_ETF_Kurs",
    "DAX_Kurs",
    "VIX_Kurs",
    "BrentOil_Kurs",
    "WTIOil_Kurs",
    "EUR_USD_Kurs",
    "EUR_CHF_Kurs",
    "EUR_GPB_Kurs",
    "EUR_TRY_Kurs",
    "Lufthansa_Kurs",
    "Ryanair_Kurs",
    "EasyJet_Kurs",
    "Air France-KLM_Kurs",
    "Booking_Kurs",
    "Expedia_Kurs",
    "Uber_Kurs",
    "Airbnb_Kurs",
    "PrimeEnergyOil_Gas_Kurs",
    "Marriott_Hotels_Kurs",
    "Hilton_Hotels_Kurs"
]

print("Berechne 1-Tages- und 5-Tages-Returns...")
df_final_returns = create_return_features(df, return_columns)

# ---------------------------------------------------------
# FINALES SPEICHERN
# ---------------------------------------------------------
# Pfad zur Ausgabe-Datei (wird ebenfalls im selben Ordner gespeichert)
OUTPUT_PATH = SCRIPT_DIR / "returns-matrix.csv"

print("Speichere die Returns-Matrix...")
df_final_returns.to_csv(OUTPUT_PATH, index=False)

print(f"Datei erfolgreich gespeichert unter: {OUTPUT_PATH}")