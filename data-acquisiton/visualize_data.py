import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ordner-Pfad zu euren Daten
DATA_DIR = r"C:\projekt\uswGruppe13\data"


def find_date_column(df):
    """Sucht automatisch nach einer Spalte, die Datumsangaben enthält."""
    for col in df.columns:
        # Wenn wir die Spalte in ein Datum umwandeln können, ohne dass es crasht, ist es die richtige!
        try:
            pd.to_datetime(df[col].iloc[:5], errors="raise")
            return col
        except:
            continue
    return None


def plot_raw_data():
    print("Erstelle Visualisierungen der Rohdaten...")
    sns.set_theme(style="whitegrid")

    # ==========================================
    # 1. AKTIENKURSE VISUALISIEREN
    # ==========================================
    stock_file = os.path.join(DATA_DIR, "EXV9_ETF.csv")
    if os.path.exists(stock_file):
        df_stock = pd.read_csv(stock_file)

        # Falls die Datei eine Kopfzeile zu viel hat (z.B. mit 'Ticker'), versuchen wir das zu flicken
        if "Ticker" in df_stock.columns or (df_stock.iloc[0] == "Ticker").any():
            # Wir lesen die Datei neu ein und überspringen die erste Zeile, falls da Müll steht
            df_stock = pd.read_csv(stock_file, skiprows=1)

        # Automatisch die Datumsspalte finden
        date_col = find_date_column(df_stock)

        if date_col:
            df_stock[date_col] = pd.to_datetime(df_stock[date_col], errors="coerce")
            # Zeilen löschen, bei denen das Datum nicht umgewandelt werden konnte
            df_stock = df_stock.dropna(subset=[date_col])

            # Suche nach einer Kurs-Spalte (Close, Last, Price oder einfach die letzte Spalte)
            close_col = None
            for c in ["Close", "Last", "Price", "Kurs"]:
                if c in df_stock.columns:
                    close_col = c
                    break
            if not close_col:
                close_col = df_stock.columns[-1]  # Fallback auf die letzte Spalte

            plt.figure(figsize=(12, 5))
            plt.plot(df_stock[date_col], pd.to_numeric(df_stock[close_col], errors="coerce"), color="blue", linewidth=2)
            plt.title("STOXX Europe 600 ETF – Verlauf (Rohdaten)", fontsize=14, fontweight="bold")
            plt.xlabel("Datum")
            plt.ylabel("Kurs")
            plt.tight_layout()
            plt.savefig(os.path.join(DATA_DIR, "plot_aktienkurs.png"))
            plt.close()
            print("✓ Diagramm für Aktienkurs gespeichert.")
        else:
            print("x Aktienkurs: Keine gültige Datumsspalte gefunden (Spalten: " + str(list(df_stock.columns)) + ")")
    else:
        print("x Aktienkurs-Datei nicht gefunden.")

    # ==========================================
    # 2. GOOGLE TRENDS VISUALISIEREN
    # ==========================================
    trends_file = os.path.join(DATA_DIR, "google_trends.csv")
    if os.path.exists(trends_file):
        df_trends = pd.read_csv(trends_file)

        date_col = find_date_column(df_trends)
        if date_col:
            df_trends[date_col] = pd.to_datetime(df_trends[date_col], errors="coerce")
            df_trends = df_trends.dropna(subset=[date_col])

            plt.figure(figsize=(12, 5))
            columns_to_plot = [col for col in df_trends.columns if col != date_col]
            for col in columns_to_plot[:2]:
                plt.plot(df_trends[date_col], pd.to_numeric(df_trends[col], errors="coerce"), label=f"Trend: {col}",
                         linewidth=2)

            plt.title("Google Trends Suchvolumen im Zeitverlauf", fontsize=14, fontweight="bold")
            plt.xlabel("Datum")
            plt.ylabel("Interesse (0 - 100)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(DATA_DIR, "plot_google_trends.png"))
            plt.close()
            print("✓ Diagramm für Google Trends gespeichert.")
        else:
            print("x Google Trends: Keine gültige Datumsspalte gefunden.")
    else:
        print("x Google Trends-Datei nicht gefunden.")

    # ==========================================
    # 3. INFLATION VISUALISIEREN
    # ==========================================
    inflation_file = os.path.join(DATA_DIR, "inflation_data.csv")
    if os.path.exists(inflation_file):
        df_inf = pd.read_csv(inflation_file)

        date_col = find_date_column(df_inf)
        if date_col:
            df_inf[date_col] = pd.to_datetime(df_inf[date_col], errors="coerce")
            df_inf = df_inf.dropna(subset=[date_col])

            rate_col = df_inf.columns[1] if len(df_inf.columns) > 1 else df_inf.columns[0]

            plt.figure(figsize=(12, 5))
            plt.step(df_inf[date_col], pd.to_numeric(df_inf[rate_col], errors="coerce"), where="post", color="red",
                     linewidth=2.5)
            plt.title("Europäische Inflationsrate (Monatliche Stufen)", fontsize=14, fontweight="bold")
            plt.xlabel("Datum")
            plt.ylabel("Inflation")
            plt.tight_layout()
            plt.savefig(os.path.join(DATA_DIR, "plot_inflation.png"))
            plt.close()
            print("✓ Diagramm für Inflation gespeichert.")
        else:
            print("x Inflation: Keine gültige Datumsspalte gefunden.")
    else:
        print("x Inflation-Datei nicht gefunden.")


if __name__ == "__main__":
    try:
        plot_raw_data()
        print("\n🎉 Alle Diagramme wurden erfolgreich im 'data'-Ordner gespeichert!")
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler bei der Visualisierung: {e}")