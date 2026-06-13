import pandas as pd

df = pd.read_csv("./finale_matrix.csv")

def create_return_features(df, columns):
    """
    Erstellt Return-Features für die angegebenen Spalten.
    """

    df_returns = pd.DataFrame()

    df_returns["Datum"] = df["Datum"]

    for col in columns:
        df_returns[f"{col}_return_1d"] = df[col].pct_change(1) * 100
        df_returns[f"{col}_return_5d"] = df[col].pct_change(5) * 100

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

df_returns = create_return_features(df, return_columns)

df_returns.to_csv(
    "finale_matrix_returns.csv",
    index=False,
    float_format="%.2f"
)
