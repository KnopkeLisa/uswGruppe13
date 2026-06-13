import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

RAW_FILE = BASE_DIR / "finale_matrix.csv"
RETURNS_FILE = BASE_DIR / "finale_matrix_returns.csv"
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
print("Entferne NaN-Werte...")
df = df.dropna()

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
