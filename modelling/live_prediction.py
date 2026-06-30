from pathlib import Path
import pandas as pd
import joblib

# ==================================================
# Projektpfade
# ==================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

LIVE_FEATURE_PATH = (
    PROJECT_ROOT
    / "data-prep-pre-split"
    / "feature_matrix_live.csv"
)

SCALER_PATH = (
    PROJECT_ROOT
    / "data-prep-post-split"
    / "feature_groups"
    / "market_core"
    / "standard_scaler.pkl"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "modelling"
    / "output_logistic_regression"
    / "logistic_regression_market_core.pkl"
)

# ==================================================
# Features definieren
# ==================================================

MARKET_CORE_FEATURES = [
    "DAX_Kurs_return_5d",
    "VIX_Kurs_return_5d",
    "BrentOil_Kurs_return_5d",
    "WTIOil_Kurs_return_5d"
]

# ==================================================
# Daten laden
# ==================================================

df_live = pd.read_csv(LIVE_FEATURE_PATH)

df_live["Datum"] = pd.to_datetime(df_live["Datum"])
df_live = df_live.sort_values("Datum")

latest_row = df_live.tail(1).copy()
latest_date = latest_row["Datum"].iloc[0]

X_live = latest_row[MARKET_CORE_FEATURES]

# ==================================================
# Scaler und Modell laden
# ==================================================

scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)

X_live_scaled = X_live.copy()
X_live_scaled[MARKET_CORE_FEATURES] = scaler.transform(
    X_live[MARKET_CORE_FEATURES]
)

# ==================================================
# Prognose
# ==================================================

prediction = model.predict(X_live_scaled)[0]
probability_positive = model.predict_proba(X_live_scaled)[0][1]

forecast_start = latest_date + pd.offsets.BDay(1)
forecast_end = latest_date + pd.offsets.BDay(5)

print()
print("=== Live Prediction ===")
print()
print("Datum der Eingangsdaten:", latest_date.date())
print("Prognosezeitraum:", forecast_start.date(), "bis", forecast_end.date())
print()
print("Vorhersage:", "Positiver Trend" if prediction == 1 else "Negativer Trend")
print("Modellwahrscheinlichkeit positiver Trend:", round(probability_positive * 100, 2), "%")