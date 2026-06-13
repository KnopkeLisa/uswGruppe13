import pandas as pd

print("Lade Return-Matrix...")

df = pd.read_csv("finale_matrix_returns.csv")

# ---------------------------------------------------------
# SPALTEN FÜR MOVING AVERAGES
# ---------------------------------------------------------

kurs_spalten = [
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

# ---------------------------------------------------------
# MOVING AVERAGES
# ---------------------------------------------------------

print("Erzeuge Moving Averages...")

for col in kurs_spalten:

    if col in df.columns:

        df[f"{col}_ma_5"] = (
            df[col]
            .rolling(5)
            .mean()
        )

        df[f"{col}_ma_20"] = (
            df[col]
            .rolling(20)
            .mean()
        )

# ---------------------------------------------------------
# TARGET VARIABLE
# ---------------------------------------------------------

print("Erzeuge Target Variable...")

df["target_return_5d"] = (
    (
        df["EXV9_ETF_Kurs"].shift(-5)
        / df["EXV9_ETF_Kurs"]
    ) - 1
) * 100

df["target_trend_5d"] = (
    df["target_return_5d"] > 0
).astype(int)

# ---------------------------------------------------------
# NAN ENTFERNEN
# ---------------------------------------------------------

print("Entferne NaN-Werte...")

df = df.dropna()

# ---------------------------------------------------------
# SPEICHERN
# ---------------------------------------------------------

output_file = "feature_matrix.csv"

df.to_csv(
    output_file,
    index=False,
    float_format="%.2f"
)

print(f"Feature Matrix gespeichert: {output_file}")
print(f"Zeilen: {len(df)}")
print(f"Spalten: {len(df.columns)}")