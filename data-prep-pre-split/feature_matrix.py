import pandas as pd
from pathlib import Path

# BASE_DIR ohne .resolve(), damit es konsistent zu den anderen Skripten bleibt
BASE_DIR = Path(__file__).parent

# PFADE KORRIGIERT: returns-matrix.csv hat jetzt den richtigen Bindestrich!
RAW_FILE = BASE_DIR / "raw-data-matrix.csv"
RETURNS_FILE = BASE_DIR / "returns-matrix.csv"  # <-- Hier war der Unterstrich-Fehler
OUTPUT_FILE = BASE_DIR / "feature_matrix.csv"

print("Lade Rohdaten-Matrix...")
df_raw = pd.read_csv(RAW_FILE)

print("Lade Return-Matrix...")
df_returns = pd.read_csv(RETURNS_FILE)

# Datum vereinheitlichen
df_raw["Datum"] = pd.to_datetime(df_raw["Datum"])
df_returns["Datum"] = pd.to_datetime(df_returns["Datum"])

# Rohdaten + Returns zusammenführen
print("Merge Rohdaten + Return-Features...")
df = pd.merge(
    df_raw,
    df_returns,
    on="Datum",
    how="inner"
)

# Target erstellen: EXV9-Rendite in 5 Handelstagen
print("Erzeuge Target Variable...")

df["target_return_5d"] = (
    (df["EXV9_ETF_Kurs"].shift(-5) / df["EXV9_ETF_Kurs"]) - 1
) * 100

# Klassifikations-Target: steigt / fällt
df["target_trend_5d"] = (
    df["target_return_5d"] > 0
).astype(int)

# Optional: Datum wieder schöner speichern
df["Datum"] = df["Datum"].dt.date

# NaN entfernen
required_cols = [
    "Datum",
    "target_return_5d",
    "target_trend_5d"
]

df = df.dropna(subset=required_cols)
return_cols = [
    col for col in df.columns
    if "_return_" in col
]

df = df.dropna(subset=return_cols)

# Speichern
df.to_csv(
    OUTPUT_FILE,
    index=False,
    float_format="%.2f"
)

print("Feature Matrix erfolgreich erstellt.")
print(f"Datei: {OUTPUT_FILE}")
print(f"Zeilen: {df.shape[0]}")
print(f"Spalten: {df.shape[1]}")