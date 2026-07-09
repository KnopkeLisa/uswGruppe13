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

MODEL_NAME = "random_forest"
GROUP = "market_core"

SCALER_PATH = PROJECT_ROOT / "data-prep-post-split" / "feature_groups" / GROUP / "standard_scaler.pkl"
MODEL_PATH = PROJECT_ROOT / "modelling" / f"output_{MODEL_NAME}" / "rf_model.pkl"

ALPACA_API_KEY = "PKPSUIBN4QOSVSDMT63QZYX664"
ALPACA_SECRET_KEY = "Gi5Zo3qNwLjRyzQLc9z9VwfKhLxaBfC6zvX2KDQFrZkM"

TRADING_SYMBOL = "SPY"
ORDER_QTY = 10
CONFIDENCE_THRESHOLD = 0.60

# ==================================================
# 1. INIT & DATEN LADEN
# ==================================================
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
    exit()

X_live_scaled = pd.DataFrame(
    scaler.transform(X_live_raw),
    columns=X_live_raw.columns,
    index=X_live_raw.index
)

# ==================================================
# 2. VORHERSAGE GENERIEREN
# ==================================================
prediction = model.predict(X_live_scaled)[0]
probability = model.predict_proba(X_live_scaled)[0][1]

print("\n" + "=" * 50)
print(f"=== LIVE TRADING SIGNAL FÜR {X_live_raw.index[0].strftime('%Y-%m-%d')} ===")
print("=" * 50)
print(f"Modell-Vorhersage: {'STEIGT (1)' if prediction == 1 else 'FÄLLT (0)'}")
print(f"Konfidenz (Wahrscheinlichkeit): {probability:.2%}")
print("-" * 50)

# ==================================================
# 3. ALPACA CLIENT & POSITIONS-CHECK
# ==================================================
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

try:
    position = trading_client.get_open_position(TRADING_SYMBOL)
    current_qty = float(position.qty)
    print(f"Aktuelle Position: Wir halten {current_qty} Stück von {TRADING_SYMBOL}.")
except Exception:
    current_qty = 0
    print(f"Aktuelle Position: Wir halten derzeit kein {TRADING_SYMBOL}.")

print("-" * 50)

# ==================================================
# 4. TRADING LOGIK (Die Strategie)
# ==================================================
if prediction == 1 and probability >= CONFIDENCE_THRESHOLD:
    if current_qty == 0:
        print(f">>> AKTION: Starkes Kaufsignal! Sende Order an Alpaca...")

        market_order_data = MarketOrderRequest(
            symbol=TRADING_SYMBOL,
            qty=ORDER_QTY,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC
        )

        trading_client.submit_order(order_data=market_order_data)
        print(f"ERFOLG: Market-Buy Order über {ORDER_QTY} {TRADING_SYMBOL} platziert.")
    else:
        print(f">>> AKTION: Wir halten {TRADING_SYMBOL} bereits. Trend ist weiterhin positiv. Kein Handlungsbedarf.")

elif prediction == 0:
    if current_qty > 0:
        print(f">>> AKTION: Trend dreht ins Negative! Verkaufe Position...")
        trading_client.close_position(TRADING_SYMBOL)
        print(f"ERFOLG: Gesamte {TRADING_SYMBOL} Position zum Verkauf freigegeben.")
    else:
        print(f">>> AKTION: Trend ist negativ, aber wir sind bereits flat (keine Aktien). Kein Handlungsbedarf.")

else:
    print(
        f">>> AKTION: Modell sagt leicht positiv, aber Konfidenz ({probability:.2%}) ist unter dem Schwellenwert ({CONFIDENCE_THRESHOLD:.2%}). Wir warten ab.")

print("=" * 50 + "\n")