import pandas as pd
import yfinance as yf
from pathlib import Path


def get_live_market_core_features(multiply_by_100=False):

    tickers = {
        "^GDAXI": "DAX_Kurs_return_5d",
        "^VIX": "VIX_Kurs_return_5d",
        "BZ=F": "BrentOil_Kurs_return_5d",
        "CL=F": "WTIOil_Kurs_return_5d"
    }

    live_features = {}
    latest_trading_date = None

    print("Starte Live-Datenabruf von Yahoo Finance...")

    for ticker, feature_name in tickers.items():
        df = yf.download(ticker, period="1mo", progress=False)

        if df.empty:
            raise ValueError(f"Kritischer Fehler: Keine Daten für Ticker {ticker} empfangen!")

        # Bereinigung:
        close_series = df['Close'].squeeze()

        # Berechnung der 5-Tages-Rendite
        returns_5d = close_series.pct_change(5)

        # Anpassung an das Datenformat
        if multiply_by_100:
            returns_5d = returns_5d * 100.0

        live_features[feature_name] = float(returns_5d.iloc[-1])

        if latest_trading_date is None:
            latest_trading_date = returns_5d.index[-1]

    live_df = pd.DataFrame([live_features], index=[latest_trading_date])
    live_df.index.name = "Datum"

    exact_column_order = [
        "DAX_Kurs_return_5d",
        "VIX_Kurs_return_5d",
        "BrentOil_Kurs_return_5d",
        "WTIOil_Kurs_return_5d"
    ]
    live_df = live_df[exact_column_order]

    return live_df


# ==================================================
# Lokaler Schnelltest
# ==================================================
if __name__ == "__main__":
    print("=== TESTE LIVE-DATA-FETCHER ===")
    try:

        df_live_test = get_live_market_core_features(multiply_by_100=False)

        print("\nErfolgreich generierter Live-Datenpunkt:")
        print(df_live_test.to_string())
        print(f"\nFormfaktor (Shape): {df_live_test.shape} -> Bereit für das Modell!")

    except Exception as e:
        print(f"\nFehler beim Testlauf aufgetreten: {e}")