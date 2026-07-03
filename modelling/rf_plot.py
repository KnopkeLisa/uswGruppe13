from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay

# ==================================================
# Pfade
# ==================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_OUTPUT_DIR = PROJECT_ROOT / "modelling" / "output_random_forest"
PLOT_OUTPUT_DIR = PROJECT_ROOT / "modelling" / "plots"
PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")


# ==================================================
# Plot-Funktionen
# ==================================================
def plot_confusion_matrix(df, title, output_name):
    cm = confusion_matrix(df["Actual"], df["Prediction"])
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fällt", "Steigt"])
    disp.plot(cmap="Blues", values_format="d", ax=plt.gca())
    plt.title(title)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_DIR / output_name, dpi=300)
    plt.close()


def plot_model_metrics(df, title, output_name):
    metrics = {
        "Accuracy": accuracy_score(df["Actual"], df["Prediction"]),
        "F1-Score": f1_score(df["Actual"], df["Prediction"], zero_division=0)
    }
    plt.figure(figsize=(6, 5))
    bars = plt.bar(metrics.keys(), metrics.values(), color=["#4c72b0", "#55a868"])
    plt.ylim(0, 1)
    plt.title(title)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{bar.get_height():.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_DIR / output_name, dpi=300)
    plt.close()


def plot_feature_importance():
    importance_file = MODEL_OUTPUT_DIR / "rf_feature_importances.csv"
    if not importance_file.exists(): return

    df_imp = pd.read_csv(importance_file).sort_values("Importance", ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=df_imp, hue="Feature", palette="viridis", legend=False)
    plt.title("Random Forest: Wichtigste Features")
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_DIR / "rf_feature_importances.png", dpi=300)
    plt.close()


# ==================================================
# Main
# ==================================================
if __name__ == "__main__":
    pred_file = MODEL_OUTPUT_DIR / "rf_test_predictions.csv"
    if pred_file.exists():
        df_preds = pd.read_csv(pred_file, index_col=0)

        plot_confusion_matrix(df_preds, "Confusion Matrix (Random Forest)", "rf_confusion_matrix.png")
        plot_model_metrics(df_preds, "Modell Metriken", "rf_model_metrics.png")
        plot_feature_importance()

        print(f"Erfolg! Plots wurden in '{PLOT_OUTPUT_DIR}' erstellt.")
    else:
        print("Fehler: Datei 'rf_test_predictions.csv' nicht gefunden. Bitte erst random_forest.py ausführen!")