import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ==================================================
# Projektpfade
# ==================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

GROUP = "market_core"
DATA_DIR = PROJECT_ROOT / "data-prep-post-split" / "feature_groups" / GROUP
OUTPUT_DIR = PROJECT_ROOT / "modelling" / "output_random_forest"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# Daten laden
# ==================================================
X_train = pd.read_csv(DATA_DIR / "X_train.csv", index_col=0)
X_val = pd.read_csv(DATA_DIR / "X_val.csv", index_col=0)
X_test = pd.read_csv(DATA_DIR / "X_test.csv", index_col=0)

y_train = pd.read_csv(DATA_DIR / "y_train.csv", index_col=0).squeeze()
y_val = pd.read_csv(DATA_DIR / "y_val.csv", index_col=0).squeeze()
y_test = pd.read_csv(DATA_DIR / "y_test.csv", index_col=0).squeeze()

# ==================================================
# Training
# ==================================================
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train_val, y_train_val)

# ==================================================
# Test-Auswertung
# ==================================================
y_test_pred = rf_model.predict(X_test)

print(f"=== Random Forest auf {GROUP} ===")
print("Accuracy auf Testdaten:", round(accuracy_score(y_test, y_test_pred), 4))
print(classification_report(y_test, y_test_pred))

pd.DataFrame({"Actual": y_test, "Prediction": y_test_pred}, index=X_test.index).to_csv(OUTPUT_DIR / "rf_test_predictions.csv")
joblib.dump(rf_model, OUTPUT_DIR / "rf_model.pkl")


return_col = [c for c in X_test.columns if 'return_5d' in c]
if return_col:
    target_return_col = return_col[0]

    actual_returns = X_test[target_return_col].copy()
    if actual_returns.abs().max() > 1.0:
        actual_returns = actual_returns / 100.0

    test_data = pd.DataFrame({
        "Actual_Return": actual_returns,
        "Signal": y_test_pred
    }, index=y_test.index)

    test_data["Strategy_Return"] = test_data["Signal"] * test_data["Actual_Return"]

    test_data["Cumulative_Strategy"] = (1 + test_data["Strategy_Return"]).cumprod()
    test_data["Cumulative_ETF"] = (1 + test_data["Actual_Return"]).cumprod()

    print(f"Endwert Strategie (Schnell-Check ohne Gebühren): {test_data['Cumulative_Strategy'].iloc[-1]:.4f}")
    print(f"Endwert ETF (Buy&Hold): {test_data['Cumulative_ETF'].iloc[-1]:.4f}")

    import matplotlib.pyplot as plt

    test_data[["Cumulative_Strategy", "Cumulative_ETF"]].plot(figsize=(10, 6))
    plt.title(f"Performance-Vergleich ({GROUP}): Modell-Strategie vs. Buy & Hold")
    plt.savefig(OUTPUT_DIR / "performance_plot.png")
    print("Plot gespeichert unter: performance_plot.png")