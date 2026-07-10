import pandas as pd
from pathlib import Path
import joblib

# ==================================================
# Setup
# ==================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "data-prep-post-split" / "feature_groups" / "market_core"
MODEL_DIR = PROJECT_ROOT / "modelling" / "output_logistic_regression"

MARKET_CORE_FEATURES = [
    "DAX_Kurs_return_5d",
    "VIX_Kurs_return_5d",
    "BrentOil_Kurs_return_5d",
    "WTIOil_Kurs_return_5d"
]

# ==================================================
# Live-Simulator Engine
# ==================================================
print("Initialisiere Paper-Trading Simulator für die Logistische Regression...")

X_test = pd.read_csv(DATA_DIR / "X_test.csv", index_col=0)
model = joblib.load(MODEL_DIR / "logistic_regression_market_core.pkl")

live_data_point = X_test[MARKET_CORE_FEATURES].tail(1)
live_date = live_data_point.index[0]

prediction = model.predict(live_data_point)[0]
probabilities = model.predict_proba(live_data_point)[0]
prob_up = probabilities[1]

print("\n" + "="*50)
print(f"=== LIVE PAPER TRADING SIGNAL (LOG REG) FÜR {live_date} ===")
print("="*50)

if prediction == 1:
    trend_str = "STEIGEND (UPTREND)"
    action_str = "KAUFEN / HALTEN (LONG POSITION)"
else:
    trend_str = "FALLEND (DOWNTREND)"
    action_str = "VERKAUFEN / CASH (FLAT POSITION)"

print(f"Modell-Prognose (nächste 5 Tage): {trend_str}")
print(f"Konfidenz (Wahrscheinlichkeit):   {prob_up:.2%}")
print("-"*50)
print(f"Empfohlene Trading-Aktion:        {action_str}")
print("Hinweis: Bei Signalwechsel fällt eine Gebühr von 0.1% an.")
print("="*50)

print("\nModell-Treiber (Gewichtung der Koeffizienten):")
try:
    for feat, coef_val in zip(MARKET_CORE_FEATURES, model.coef_[0]):
        current_val = live_data_point[feat].iloc[0]
        print(f"- {feat}: {current_val:.4f} (Koeffizient: {coef_val:.4f})")
except Exception as e:
    print(f"Treiber konnten nicht visualisiert werden: {e}")