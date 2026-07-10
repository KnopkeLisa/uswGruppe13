import pandas as pd
from pathlib import Path
import joblib

# ==================================================
# Setup
# ==================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

GROUP = "market_core"
DATA_DIR = PROJECT_ROOT / "data-prep-post-split" / "feature_groups" / GROUP
MODEL_DIR = PROJECT_ROOT / "modelling" / "output_random_forest"

# ==================================================
# Live-Simulator Engine
# ==================================================
print(f"Initialisiere Paper-Trading Simulator mit Gruppe: {GROUP}...")

X_test = pd.read_csv(DATA_DIR / "X_test.csv", index_col=0)
rf_model = joblib.load(MODEL_DIR / "rf_model.pkl")

live_data_point = X_test.iloc[[-1]]
live_date = live_data_point.index[0]

prediction = rf_model.predict(live_data_point)[0]
probabilities = rf_model.predict_proba(live_data_point)[0]

prob_up = probabilities[1]

# ==================================================
# Trading Logik
# ==================================================
print("\n" + "="*50)
print(f"=== LIVE PAPER TRADING SIGNAL FÜR {live_date} ===")
print("="*50)

if prediction == 1:
    trend_str = "STEIGEND (UPTREND)"
    action_str = "KAUFEN / HALTEN (LONG POSITION)"
else:
    trend_str = "FALLEND (DOWNTREND)"
    action_str = "VERKAUFEN / CASH (FLAT POSITION)"

print(f"Modell-Prognose (nächste 5 Tage): {trend_str}")
print(f"Konfidenz (Wahrscheinlichkeit):   {prob_up:.2%}")
print("-" * 50)
print(f"Empfohlene Trading-Aktion:        {action_str}")
print("="*50)

feature_importances = pd.Series(rf_model.feature_importances_, index=X_test.columns)
top_3_features = feature_importances.nlargest(3)

print("\nTreiber für diese Entscheidung (Top 3 Features im Modell):")
for feat, imp in top_3_features.items():
    current_val = live_data_point[feat].iloc[0]
    print(f"- {feat}: {current_val:.4f} (Gewichtung: {imp:.2%})")