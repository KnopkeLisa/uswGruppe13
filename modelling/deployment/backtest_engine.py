import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
PRE_SPLIT_PATH = PROJECT_ROOT / "data-prep-pre-split" / "feature_matrix.csv"

# ==================================================
# Daten & Modell
# ==================================================
print(f"Lade Daten ({GROUP}) und Random Forest für den Backtest...")
X_test = pd.read_csv(DATA_DIR / "X_test.csv", index_col=0)
y_test = pd.read_csv(DATA_DIR / "y_test.csv", index_col=0).squeeze()
rf_model = joblib.load(MODEL_DIR / "rf_model.pkl")

X_test.index = pd.to_datetime(X_test.index)
y_test.index = pd.to_datetime(y_test.index)

y_test_pred = rf_model.predict(X_test)

# ==================================================
# ETF-Rendite
# ==================================================
df_orig = pd.read_csv(PRE_SPLIT_PATH)
if "Datum" in df_orig.columns:
    df_orig["Datum"] = pd.to_datetime(df_orig["Datum"])
    df_orig = df_orig.set_index("Datum")
else:
    df_orig.index = pd.to_datetime(df_orig.index)

df_orig = df_orig.sort_index()

etf_return_cols = [c for c in df_orig.columns if "target_return" in c or ("EXV9" in c and "return_5d" in c)]
if not etf_return_cols:
    etf_return_cols = [c for c in df_orig.columns if "return_5d" in c]

target_return_col = etf_return_cols[0]
actual_returns = df_orig.loc[X_test.index, target_return_col].copy()

if actual_returns.abs().max() > 1.0:
    print(f"-> Info: Prozentwerte erkannt ({target_return_col}). Konvertiere in Dezimalzahlen...")
    actual_returns = actual_returns / 100.0

# ==================================================
# Backtest Engine
# ==================================================
bt_df = pd.DataFrame({
    "Actual_Return": actual_returns,
    "Signal": y_test_pred
}, index=X_test.index)

bt_df["Strategy_Return_Raw"] = bt_df["Signal"] * bt_df["Actual_Return"]

bt_df["Signal_Change"] = bt_df["Signal"].diff().abs().fillna(0)

if len(bt_df) > 0 and bt_df["Signal"].iloc[0] == 1:
    bt_df.iloc[0, bt_df.columns.get_loc("Signal_Change")] = 1.0

HANDELSGEBUEHR = 0.0
total_trades = int(bt_df["Signal_Change"].sum())

bt_df["Strategy_Return"] = bt_df["Strategy_Return_Raw"] - (bt_df["Signal_Change"] * HANDELSGEBUEHR)

bt_df["Cumulative_Strategy"] = (1 + bt_df["Strategy_Return"]).cumprod()
bt_df["Cumulative_Hold"] = (1 + bt_df["Actual_Return"]).cumprod()

# ==================================================
# Kennzahlen-Berechnung
# ==================================================
win_rate = (bt_df[bt_df["Signal"] == 1]["Strategy_Return"] > 0).mean()
roll_max = bt_df["Cumulative_Strategy"].cummax()
drawdown = bt_df["Cumulative_Strategy"] / roll_max - 1.0
max_drawdown = drawdown.min()

print("\n" + "="*40)
print("=== RANDOM FOREST BACKTEST ERGEBNISSE (MIT GEBÜHREN) ===")
print("="*40)
print(f"Genutzte Feature-Gruppe        : {GROUP}")
print(f"Eingerechnete Gebühr pro Trade : {HANDELSGEBUEHR * 100}%")
print(f"Anzahl durchgeführter Trades   : {total_trades}")
print(f"Endwert Strategie (Netto)      : {bt_df['Cumulative_Strategy'].iloc[-1]:.4f}")
print(f"Endwert Buy & Hold (Benchmark) : {bt_df['Cumulative_Hold'].iloc[-1]:.4f}")
print(f"Win Rate (Investierte Tage)    : {win_rate:.2%}")
print(f"Max Drawdown Strategie         : {max_drawdown:.2%}")
print("="*40)

# ==================================================
# Plot generieren
# ==================================================
plt.figure(figsize=(12, 6))
plt.plot(bt_df.index, bt_df["Cumulative_Strategy"], label="RandomForest-Strategie (Netto inkl. Gebühren)", color="blue")
plt.plot(bt_df.index, bt_df["Cumulative_Hold"], label="Buy & Hold (Benchmark)", color="gray", alpha=0.7)
plt.title(f"Backtest: Random Forest ({GROUP}) vs. Benchmark (Inkl. Kosten)")
plt.ylabel("Kumulativer Wert (Start = 1.0)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(MODEL_DIR / "advanced_backtest_plot_with_fees.png")
print(f"Plot erfolgreich gespeichert unter:\n{MODEL_DIR / 'advanced_backtest_plot_with_fees.png'}")