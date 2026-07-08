import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

# ==================================================
# Setup & Pfade
# ==================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data-prep-post-split" / "feature_groups" / "market_core"
MODEL_DIR = PROJECT_ROOT / "modelling" / "output_logistic_regression"

# ==================================================
# Daten & Modell laden
# ==================================================
print("Lade Daten für den Robustheits-Check (Logistische Regression -> Peer-Group)...")
X_test = pd.read_csv(DATA_DIR / "X_test.csv", index_col=0)
X_test.index = pd.to_datetime(X_test.index)
model = joblib.load(MODEL_DIR / "logistic_regression_market_core.pkl")

# Signale generieren
y_test_pred = model.predict(X_test)

# Robustheits-Test auf Lufthansa
peer_assets = [c for c in X_test.columns if 'Lufthansa' in c and 'return_5d' in c]

if not peer_assets:
    PRE_SPLIT_PATH = PROJECT_ROOT / "data-prep-pre-split" / "feature_matrix.csv"
    df_orig = pd.read_csv(PRE_SPLIT_PATH, index_col=0)
    df_orig.index = pd.to_datetime(df_orig.index)
    peer_assets = [c for c in df_orig.columns if 'Lufthansa' in c and 'return_5d' in c]
    if peer_assets:
        peer_col = peer_assets[0]
        peer_returns = df_orig.loc[X_test.index, peer_col].copy()
    else:
        peer_returns = None
else:
    peer_col = peer_assets[0]
    peer_returns = X_test[peer_col].copy()

if peer_returns is None:
    print("Fehler: Keine Lufthansa-Daten im Projekt gefunden. Bitte Spalten prüfen.")
else:
    if peer_returns.abs().max() > 1.0:
        print(f"-> Info: Prozentwerte erkannt ({peer_col}). Konvertiere in Dezimalzahlen...")
        peer_returns = peer_returns / 100.0

    rob_df = pd.DataFrame({
        "Peer_Return": peer_returns,
        "Signal": y_test_pred
    }, index=X_test.index)

    rob_df["Strategy_Return_Raw"] = rob_df["Signal"] * rob_df["Peer_Return"]
    rob_df["Signal_Change"] = rob_df["Signal"].diff().abs().fillna(0)

    if len(rob_df) > 0 and rob_df["Signal"].iloc[0] == 1:
        rob_df.iloc[0, rob_df.columns.get_loc("Signal_Change")] = 1.0

    HANDELSGEBUEHR = 0.001
    rob_df["Strategy_Return"] = rob_df["Strategy_Return_Raw"] - (rob_df["Signal_Change"] * HANDELSGEBUEHR)

    rob_df["Cumulative_Strategy"] = (1 + rob_df["Strategy_Return"]).cumprod()
    rob_df["Cumulative_Peer_Hold"] = (1 + rob_df["Peer_Return"]).cumprod()

    print("\n" + "=" * 40)
    print("=== LOGREG ROBUSTNESS CHECK: Lufthansa (MIT GEBÜHREN) ===")
    print("=" * 40)
    print(f"Endwert Strategie auf Peer: {rob_df['Cumulative_Strategy'].iloc[-1]:.4f}")
    print(f"Endwert Buy & Hold Peer:    {rob_df['Cumulative_Peer_Hold'].iloc[-1]:.4f}")
    print("=" * 40)

    plt.figure(figsize=(10, 5))
    plt.plot(rob_df.index, rob_df["Cumulative_Strategy"], label="LogReg-Strategie (auf Lufthansa)", color="orange")
    plt.plot(rob_df.index, rob_df["Cumulative_Peer_Hold"], label="Buy & Hold (Lufthansa)", color="gray", alpha=0.7)
    plt.title("Robustheits-Test LogReg: Transfer auf Lufthansa-Aktie (inkl. Gebühren)")
    plt.ylabel("Kumulativer Wert (Start = 1.0)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(MODEL_DIR / "robustness_peer_test_with_fees.png")
    print(f"Plot gespeichert unter: {MODEL_DIR / 'robustness_peer_test_with_fees.png'}")