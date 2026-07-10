import pandas as pd
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
# Daten laden
# ==================================================
print(f"Lade Daten ({GROUP}) für den Robustheits-Check (Peer-Group)...")
X_test = pd.read_csv(DATA_DIR / "X_test.csv", index_col=0)
X_test.index = pd.to_datetime(X_test.index)

rf_model = joblib.load(MODEL_DIR / "rf_model.pkl")
y_test_pred = rf_model.predict(X_test)

# ==================================================
# Robustheits-Test
# ==================================================
print("Lade Lufthansa-Peer-Daten aus der Feature-Matrix...")
df_orig = pd.read_csv(PRE_SPLIT_PATH)
if "Datum" in df_orig.columns:
    df_orig["Datum"] = pd.to_datetime(df_orig["Datum"])
    df_orig = df_orig.set_index("Datum")
else:
    df_orig.index = pd.to_datetime(df_orig.index)

df_orig = df_orig.sort_index()

peer_assets = [c for c in df_orig.columns if 'Lufthansa' in c and 'return_5d' in c]

if not peer_assets:
    print("Keine Lufthansa-Daten in der Feature-Matrix gefunden. Bitte Spaltennamen prüfen.")
else:
    peer_col = peer_assets[0]
    peer_returns = df_orig.loc[X_test.index, peer_col].copy()

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

    HANDELSGEBUEHR = 0.0
    total_trades = int(rob_df["Signal_Change"].sum())

    rob_df["Strategy_Return"] = rob_df["Strategy_Return_Raw"] - (rob_df["Signal_Change"] * HANDELSGEBUEHR)


    rob_df["Cumulative_Strategy"] = (1 + rob_df["Strategy_Return"]).cumprod()
    rob_df["Cumulative_Peer_Hold"] = (1 + rob_df["Peer_Return"]).cumprod()

    print("\n" + "=" * 40)
    print(f"=== KORRIGIERTER ROBUSTNESS CHECK: Lufthansa ===")
    print("=" * 40)
    print("Hypothese: Die gelernten Signale des ETF-Modells lassen sich")
    print("auf peer-verwandte Tourismus-Werte übertragen.")
    print("-" * 40)
    print(f"Eingerechnete Gebühr pro Trade : {HANDELSGEBUEHR * 100}%")
    print(f"Anzahl durchgeführter Trades   : {total_trades}")
    print(f"Endwert Strategie auf Peer (Netto): {rob_df['Cumulative_Strategy'].iloc[-1]:.4f}")
    print(f"Endwert Buy & Hold Peer:           {rob_df['Cumulative_Peer_Hold'].iloc[-1]:.4f}")
    print("=" * 40)


    plt.figure(figsize=(10, 5))
    plt.plot(rob_df.index, rob_df["Cumulative_Strategy"], label="Modell-Strategie (auf Lufthansa, Netto)",
             color="orange")
    plt.plot(rob_df.index, rob_df["Cumulative_Peer_Hold"], label="Buy & Hold (Lufthansa)", color="gray", alpha=0.7)
    plt.title("Robustheits-Test: Transfer auf Lufthansa-Aktie (Inkl. Gebühren)")
    plt.ylabel("Kumulativer Wert (Start = 1.0)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(MODEL_DIR / "robustness_peer_test.png")
    print(f"Plot gespeichert unter: {MODEL_DIR / 'robustness_peer_test.png'}")