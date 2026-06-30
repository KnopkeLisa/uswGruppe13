import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

RAW_FILE = BASE_DIR / "raw-data-matrix.csv"
RETURNS_FILE = BASE_DIR / "returns_matrix.csv"
OUTPUT_FILE = BASE_DIR / "feature_matrix_live.csv"

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

df = df.sort_values("Datum")

# KEIN Target erzeugen und KEINE letzten 5 Tage entfernen
# Diese Matrix ist nur für Live Prediction gedacht.

return_cols = [
    col for col in df.columns
    if "_return_" in col
]

# Nur Zeilen entfernen, bei denen Return-Features fehlen
df = df.dropna(subset=return_cols)

# Datum schöner speichern
df["Datum"] = df["Datum"].dt.date

df.to_csv(
    OUTPUT_FILE,
    index=False,
    float_format="%.2f"
)

print()
print("Live Feature Matrix erfolgreich erstellt.")
print(f"Datei: {OUTPUT_FILE}")
print(f"Zeilen: {df.shape[0]}")
print(f"Spalten: {df.shape[1]}")
print(f"Von: {df['Datum'].min()}")
print(f"Bis: {df['Datum'].max()}")