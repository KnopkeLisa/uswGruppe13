import os
import joblib
import pandas as pd
from pathlib import Path

# Alpaca SDK Komponenten
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from live_data_fetcher import get_live_market_core_features

# ==================================================
# CONFIGURATION
# ==================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# HIER DAS MODELL WECHSELN:
MODEL_NAME = "random_forest"

GROUP = "market_core"

SCALER_PATH = PROJECT_ROOT / "data-prep-post-split" / "feature_groups" / GROUP / "standard_scaler.pkl"
MODEL_PATH = PROJECT_ROOT / "modelling" / f"output_{MODEL_NAME}" / "rf_model.pkl"

ALPACA_API_KEY = "PKPSUIBN4QOSVSDMT63QZYX664"
ALPACA_SECRET_KEY = "Gi5Zo3qNwLjRyzQLc9z9VwfKhLxaBfC6zvX2KDQFrZkM"

TRADING_SYMBOL = "SPY"
ORDER_QTY = 10

# ==================================================
# MAIN TRADING ENGINE
# ==================================================
def run_live_trader():
    print("\n==================================================")
    print(f"STARTING LIVE TRADER ENGINE (MODE: {MODEL_NAME.upper()})")
    print("==================================================")

    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Fehler: Modell '{MODEL_NAME}' nicht gefunden unter: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError("Fehler: standard_scaler.pkl wurde nicht gefunden!")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("-> Modell und zugehörigen StandardScaler erfolgreich geladen.")

    try:
        X_live_raw = get_live_market_core_features(multiply_by_100=False)
        print(f"-> Live-Rohdaten abgerufen für: {X_live_raw.index[0].strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"!!! KRITISCHER FEHLER beim Datenabruf: {e}")
        return

    X_live_scaled = pd.DataFrame(
        scaler.transform(X_live_raw),
        columns=X_live_raw.columns,
        index=X_live_raw.index
    )
    print(f"-> Live-Daten erfolgreich mit dem Trainings-Scaler transformiert.")

    prediction = int(model.predict(X_live_scaled)[0])
    probabilities = model.predict_proba(X_live_scaled)[0]
    confidence = probabilities[prediction]

    print(f"-> Modell-Signal: {prediction} (Konfidenz: {confidence:.2%})")

    positions = trading_client.get_all_positions()
    is_invested = any(pos.symbol == TRADING_SYMBOL for pos in positions)

    print(f"-> Depot-Status bei Alpaca: {'INVESTIERT' if is_invested else 'CASH'}")
    print("--------------------------------------------------")

    if prediction == 1:
        if not is_invested:
            print(f"SIGNAL: KAUFEN. Sende Kaufauftrag für {ORDER_QTY}x {TRADING_SYMBOL}...")
            market_order_data = MarketOrderRequest(
                symbol=TRADING_SYMBOL, qty=ORDER_QTY, side=OrderSide.BUY, time_in_force=TimeInForce.DAY
            )
            trading_client.submit_order(order_data=market_order_data)
            print("-> Kaufauftrag erfolgreich übermittelt!")
        else:
            print(f"SIGNAL: HALTEN. Bereits investiert.")
    elif prediction == 0:
        if is_invested:
            print(f"SIGNAL: VERKAUFEN. Schließe Position für {TRADING_SYMBOL}...")
            trading_client.close_position(TRADING_SYMBOL)
            print("-> Position erfolgreich glattgestellt!")
        else:
            print(f"SIGNAL: CASH BLEIBEN. Keine Aktion nötig.")

    print("==================================================")
    print("LIVE TRADER RUN COMPLETED.")
    print("==================================================\n")

if __name__ == "__main__":
    run_live_trader()