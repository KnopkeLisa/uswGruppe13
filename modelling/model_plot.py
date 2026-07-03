from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)


# ==================================================
# Projektpfade
# ==================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MODEL_OUTPUT_DIR = PROJECT_ROOT / "modelling" / "output_logistic_regression"
PLOT_OUTPUT_DIR = PROJECT_ROOT / "modelling" / "plots"

PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# Predictions laden
# ==================================================

def load_predictions(filename: str) -> pd.DataFrame:
    path = MODEL_OUTPUT_DIR / filename

    df = pd.read_csv(path)

    if "Datum" in df.columns:
        df["Datum"] = pd.to_datetime(df["Datum"])
        df = df.set_index("Datum")

    return df


# ==================================================
# Plot 1: Confusion Matrix
# ==================================================

def plot_confusion_matrix(df: pd.DataFrame, title: str, output_name: str):
    y_true = df["Actual"]
    y_pred = df["Prediction"]

    plt.figure(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Fällt", "Steigt"],
        cmap="Blues",
        values_format="d"
    )

    plt.title(title)
    plt.tight_layout()

    output_path = PLOT_OUTPUT_DIR / output_name
    plt.savefig(output_path, dpi=300)
    plt.show()

    print("Plot gespeichert unter:", output_path)


# ==================================================
# Plot 2: Modellmetriken
# ==================================================

def plot_model_metrics(df: pd.DataFrame, title: str, output_name: str):
    y_true = df["Actual"]
    y_pred = df["Prediction"]

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0)
    }

    metrics_df = pd.DataFrame({
        "Metrik": list(metrics.keys()),
        "Wert": list(metrics.values())
    })

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        metrics_df["Metrik"],
        metrics_df["Wert"]
    )

    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.2f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    output_path = PLOT_OUTPUT_DIR / output_name
    plt.savefig(output_path, dpi=300)
    plt.show()

    print("Plot gespeichert unter:", output_path)


# ==================================================
# Plot 3: Wahrscheinlichkeitsplot
# ==================================================

def plot_prediction_probability(df: pd.DataFrame, title: str, output_name: str):
    plt.figure(figsize=(14, 6))

    plt.plot(
        df.index,
        df["Probability_Positive_Trend"],
        label="Modellwahrscheinlichkeit: ETF steigt",
        linewidth=2
    )

    plt.axhline(
        y=0.5,
        linestyle="--",
        label="Entscheidungsgrenze 50%"
    )

    plt.scatter(
        df.index,
        df["Actual"],
        label="Tatsächlicher Trend",
        alpha=0.5
    )

    plt.yticks(
        [0, 0.5, 1],
        ["Fällt", "50%", "Steigt"]
    )

    plt.ylim(-0.05, 1.05)
    plt.ylabel("Wahrscheinlichkeit / tatsächliche Klasse")
    plt.xlabel("Datum")
    plt.title(title)

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = PLOT_OUTPUT_DIR / output_name
    plt.savefig(output_path, dpi=300)
    plt.show()

    print("Plot gespeichert unter:", output_path)


# ==================================================
# Plot 4: Richtige vs. falsche Vorhersagen über Zeit
# ==================================================

def plot_correct_predictions_over_time(df: pd.DataFrame, title: str, output_name: str):
    plot_df = df.copy()

    plot_df["Correct"] = (
        plot_df["Actual"] == plot_df["Prediction"]
    ).astype(int)

    plt.figure(figsize=(14, 4))

    plt.scatter(
        plot_df.index,
        plot_df["Correct"],
        alpha=0.7
    )

    plt.yticks(
        [0, 1],
        ["Falsch", "Richtig"]
    )

    plt.ylabel("Vorhersage")
    plt.xlabel("Datum")
    plt.title(title)

    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = PLOT_OUTPUT_DIR / output_name
    plt.savefig(output_path, dpi=300)
    plt.show()

    print("Plot gespeichert unter:", output_path)


# ==================================================
# Alle Plots für einen Datensatz erzeugen
# ==================================================

def create_all_plots(filename: str, dataset_name: str, prefix: str):
    df = load_predictions(filename)

    print()
    print("Erzeuge Plots für:", dataset_name)
    print("Datei:", filename)
    print("Zeilen:", len(df))

    plot_confusion_matrix(
        df,
        f"Confusion Matrix: Logistic Regression auf {dataset_name}",
        f"{prefix}_confusion_matrix.png"
    )

    plot_model_metrics(
        df,
        f"Modellmetriken: Logistic Regression auf {dataset_name}",
        f"{prefix}_model_metrics.png"
    )

    plot_prediction_probability(
        df,
        f"Modellwahrscheinlichkeit: positiver EXV9-Trend auf {dataset_name}",
        f"{prefix}_prediction_probability.png"
    )

    plot_correct_predictions_over_time(
        df,
        f"Richtige und falsche Vorhersagen über die Zeit: {dataset_name}",
        f"{prefix}_correct_predictions_over_time.png"
    )


# ==================================================
# Main
# ==================================================

if __name__ == "__main__":

    # Testdaten
    create_all_plots(
        filename="test_predictions.csv",
        dataset_name="Testdaten",
        prefix="test"
    )

    # Falls vorhanden: Validation-Daten
    validation_file = MODEL_OUTPUT_DIR / "validation_predictions.csv"

    if validation_file.exists():
        create_all_plots(
            filename="validation_predictions.csv",
            dataset_name="Validation-Daten",
            prefix="validation"
        )
    else:
        print()
        print("Hinweis:")
        print("validation_predictions.csv wurde nicht gefunden.")
        print("Wenn du Validation-Plots möchtest, speichere im Modellskript zusätzlich die Validation-Predictions.")