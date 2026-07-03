from pathlib import Path
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# ==================================================
# Projektpfade
# ==================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / "data-prep-post-split" / "feature_groups" / "market_core"
OUTPUT_DIR = PROJECT_ROOT / "modelling" / "output_logistic_regression"

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
# Baseline: immer positiver Trend
# ==================================================

baseline_pred = [1] * len(y_val)

print("=== Baseline Validation ===")
print("Accuracy:", round(accuracy_score(y_val, baseline_pred), 4))
print("F1:", round(f1_score(y_val, baseline_pred), 4))
print()


# ==================================================
# Logistic Regression trainieren
# ==================================================

model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)


# ==================================================
# Validation auswerten
# ==================================================

y_val_pred = model.predict(X_val)
y_val_prob = model.predict_proba(X_val)[:, 1]

print("=== Logistic Regression Validation ===")
print("Accuracy:", round(accuracy_score(y_val, y_val_pred), 4))
print("Precision:", round(precision_score(y_val, y_val_pred), 4))
print("Recall:", round(recall_score(y_val, y_val_pred), 4))
print("F1:", round(f1_score(y_val, y_val_pred), 4))
print()
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))
print()
print(classification_report(y_val, y_val_pred))


# ==================================================
# Koeffizienten anzeigen
# ==================================================

coef_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("=== Feature-Koeffizienten ===")
print(coef_df)

coef_df.to_csv(OUTPUT_DIR / "logistic_regression_coefficients.csv", index=False)


# ==================================================
# Finales Modell: Train + Validation
# ==================================================

X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

final_model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42
)

final_model.fit(X_train_val, y_train_val)


# ==================================================
# Testdaten auswerten
# ==================================================

y_test_pred = final_model.predict(X_test)
y_test_prob = final_model.predict_proba(X_test)[:, 1]

print()
print("=== Finales Modell auf Testdaten ===")
print("Accuracy:", round(accuracy_score(y_test, y_test_pred), 4))
print("Precision:", round(precision_score(y_test, y_test_pred), 4))
print("Recall:", round(recall_score(y_test, y_test_pred), 4))
print("F1:", round(f1_score(y_test, y_test_pred), 4))
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
print()
print(classification_report(y_test, y_test_pred))


# ==================================================
# Vorhersagen speichern
# ==================================================

predictions_df = pd.DataFrame({
    "Datum": X_test.index,
    "Actual": y_test.values,
    "Prediction": y_test_pred,
    "Probability_Positive_Trend": y_test_prob
})

predictions_df.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
)

joblib.dump(
    final_model,
    OUTPUT_DIR / "logistic_regression_market_core.pkl"
)

print()
print("Vorhersagen gespeichert unter:")
print(OUTPUT_DIR / "test_predictions.csv")

print()
print("Modell gespeichert unter:")
print(OUTPUT_DIR / "logistic_regression_market_core.pkl")


# ==================================================
# Beispiel: letzte verfügbare Beobachtung vorhersagen
# ==================================================

latest_row = X_test.tail(1)
latest_prediction = final_model.predict(latest_row)[0]
latest_probability = final_model.predict_proba(latest_row)[0][1]

print()
print("=== Beispiel-Vorhersage letzte Testzeile ===")
print("Datum:", latest_row.index[0])
print("Vorhersage:", "ETF steigt" if latest_prediction == 1 else "ETF fällt")
print("Wahrscheinlichkeit positiver Trend:", round(latest_probability * 100, 2), "%")