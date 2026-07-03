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
# Wir wählen hier eine der Gruppen, die dein Kollege erstellt hat
GROUP = "all_selected"
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

# Speichern
pd.DataFrame({"Actual": y_test, "Prediction": y_test_pred}, index=X_test.index).to_csv(OUTPUT_DIR / "rf_test_predictions.csv")
joblib.dump(rf_model, OUTPUT_DIR / "rf_model.pkl")