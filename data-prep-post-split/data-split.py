from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ==================================================
# Projektpfade
# ==================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_PATH = PROJECT_ROOT / "data-prep-pre-split" / "feature_matrix.csv"
OUTPUT_DIR = PROJECT_ROOT / "data-prep-post-split" / "feature_groups"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# Daten laden
# ==================================================

df = pd.read_csv(INPUT_PATH)

if "Datum" in df.columns:
    df["Datum"] = pd.to_datetime(df["Datum"])
    df = df.set_index("Datum")
else:
    df.index = pd.to_datetime(df.index)

df = df.sort_index()


# ==================================================
# Chronologischer Split
# ==================================================

train = df.loc["2023-01-10":"2024-12-31"]
val = df.loc["2025-01-01":"2025-12-31"]
test = df.loc["2026-01-01":"2026-06-29"]


# ==================================================
# Target
# ==================================================

TARGET = "target_trend_5d"

DROP_COLUMNS = [
    "target_return_5d",
    "target_trend_5d"
]


# ==================================================
# Featuregruppen definieren
# ==================================================

feature_groups = {
    "market_core": [
        "DAX_Kurs_return_5d",
        "VIX_Kurs_return_5d",
        "BrentOil_Kurs_return_5d",
        "WTIOil_Kurs_return_5d"
    ],

    "tourism_stocks": [
        "Lufthansa_Kurs_return_5d",
        "Ryanair_Kurs_return_5d",
        "EasyJet_Kurs_return_5d",
        "Air France-KLM_Kurs_return_5d",
        "Booking_Kurs_return_5d",
        "Expedia_Kurs_return_5d",
        "Airbnb_Kurs_return_5d",
        "Marriott_Hotels_Kurs_return_5d",
        "Hilton_Hotels_Kurs_return_5d"
    ],

    "seasonal": [
        "vacation_christmas_block",
        "vacation_easter_block",
        "vacation_summer_wave",
        "vacation_autumn_wave",
        "vacation_winter_ski_wave",
        "holiday"
    ],

    "search_trends_sentiment": [
        "Mallorca",
        "wellness",
        "airbnb",
        "hotel",
        "camping"
    ],

    "macro_health_weather": [
        "Inflation_Gesamt",
        "Inflation_Restaurants_Hotels",
        "Inflation_Transport",
        "ECDC_Europa_Inzidenz",
        "CentralEurope_Temp",
        "HolidayDestination_Temp",
        "CentralEurope_Sunshine",
        "HolidayDestination_Sunshine",
        "CentralEurope_Precipitation",
        "HolidayDestination_Precipitation"
    ],

    "currency": [
        "EUR_USD_Kurs_return_5d",
        "EUR_CHF_Kurs_return_5d",
        "EUR_GPB_Kurs_return_5d",
        "EUR_TRY_Kurs_return_5d"
    ]
}


# Kombinierte Gruppen
feature_groups["market_plus_tourism"] = (
    feature_groups["market_core"]
    + feature_groups["tourism_stocks"]
)

feature_groups["market_tourism_seasonal"] = (
    feature_groups["market_core"]
    + feature_groups["tourism_stocks"]
    + feature_groups["seasonal"]
)

feature_groups["market_plus_seasonal"] = (
    feature_groups["market_core"]
    + feature_groups["seasonal"]
)

feature_groups["all_selected"] = (
    feature_groups["market_core"]
    + feature_groups["tourism_stocks"]
    + feature_groups["seasonal"]
    + feature_groups["search_trends_sentiment"]
    + feature_groups["macro_health_weather"]
    + feature_groups["currency"]
)


# ==================================================
# Bewertungsfunktion
# ==================================================

def evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0)
    }


# ==================================================
# Featuregruppen testen
# ==================================================

results = []

for group_name, features in feature_groups.items():

    print(f"\nTeste Featuregruppe: {group_name}")

    # Nur Features verwenden, die wirklich existieren
    existing_features = [
        feature for feature in features
        if feature in df.columns
    ]

    missing_features = [
        feature for feature in features
        if feature not in df.columns
    ]

    if missing_features:
        print("Fehlende Features:", missing_features)

    if len(existing_features) == 0:
        print("Übersprungen, keine Features vorhanden.")
        continue

    group_output_dir = OUTPUT_DIR / group_name
    group_output_dir.mkdir(parents=True, exist_ok=True)

    X_train = train[existing_features].copy()
    y_train = train[TARGET].copy()

    X_val = val[existing_features].copy()
    y_val = val[TARGET].copy()

    X_test = test[existing_features].copy()
    y_test = test[TARGET].copy()

    # Fehlende Werte behandeln
    X_train = X_train.ffill().bfill()
    X_val = X_val.ffill().bfill()
    X_test = X_test.ffill().bfill()

    # Skalierung nur auf Trainingsdaten fitten
    numeric_columns = X_train.select_dtypes(
        include=["float64", "int64"]
    ).columns

    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_columns] = scaler.fit_transform(
        X_train[numeric_columns]
    )

    X_val_scaled[numeric_columns] = scaler.transform(
        X_val[numeric_columns]
    )

    X_test_scaled[numeric_columns] = scaler.transform(
        X_test[numeric_columns]
    )

    # CSVs speichern
    X_train_scaled.to_csv(group_output_dir / "X_train.csv")
    X_val_scaled.to_csv(group_output_dir / "X_val.csv")
    X_test_scaled.to_csv(group_output_dir / "X_test.csv")

    y_train.to_csv(group_output_dir / "y_train.csv")
    y_val.to_csv(group_output_dir / "y_val.csv")
    y_test.to_csv(group_output_dir / "y_test.csv")

    joblib.dump(scaler, group_output_dir / "standard_scaler.pkl")

    # Modelle testen
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=4000,
            random_state=42
        )
        # "RandomForest": RandomForestClassifier(
        #     n_estimators=300,
        #     max_depth=5,
        #     random_state=42,
        #     class_weight="balanced"
        # )
    }

    for model_name, model in models.items():

        metrics = evaluate_model(
            model,
            X_train_scaled,
            y_train,
            X_val_scaled,
            y_val
        )

        results.append({
            "feature_group": group_name,
            "model": model_name,
            "num_features": len(existing_features),
            "train_rows": len(X_train_scaled),
            "val_rows": len(X_val_scaled),
            "test_rows": len(X_test_scaled),
            "val_accuracy": metrics["accuracy"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_f1": metrics["f1"],
            "features": ", ".join(existing_features)
        })


# ==================================================
# Ergebnisse speichern
# ==================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="val_f1",
    ascending=False
)

results_df.to_csv(
    OUTPUT_DIR / "feature_group_results.csv",
    index=False
)

print("\nFeaturegruppen-Test abgeschlossen.")
print("Ergebnisse gespeichert unter:")
print(OUTPUT_DIR / "feature_group_results.csv")

print("\nTop Ergebnisse:")
print(
    results_df[
        [
            "feature_group",
            "model",
            "num_features",
            "val_accuracy",
            "val_precision",
            "val_recall",
            "val_f1"
        ]
    ].head(30)
)