import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Design für die Plots hübsch und professionell machen
sns.set_theme(style="whitegrid", context="talk")

# Pfade absolut sicher auflösen
BASE_DIR = Path(__file__).parent

# Frisch gemergte Tabellen krisensicher laden (Dateiname mit Bindestrich korrigiert)
df_raw = pd.read_csv(BASE_DIR / 'raw-data-matrix.csv', index_col='Datum', parse_dates=True)
df_returns = pd.read_csv(BASE_DIR / 'returns-matrix.csv', index_col='Datum', parse_dates=True)
df_feature = pd.read_csv(BASE_DIR / 'feature_matrix.csv', index_col='Datum', parse_dates=True)

color1 = '#1f77b4' # Blau
color2 = '#d62728' # Rot
color3 = '#ff7f0e' # Orange

# ---------------------------------------------------------
# PLOT 1: Die Korrelations-Heatmap (Der absolute Prof-Liebling)
# ---------------------------------------------------------
print("Generiere Plot 1: Korrelations-Heatmap...")
plt.figure(figsize=(10, 8))

korrelationen = df_raw.corr()['EXV9_ETF_Kurs'].drop('EXV9_ETF_Kurs')
top_features = korrelationen.abs().sort_values(ascending=False).head(10).index
spalten_fuer_heatmap = ['EXV9_ETF_Kurs'] + list(top_features)

sns.heatmap(df_raw[spalten_fuer_heatmap].corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Top 10 Zusammenhänge: Was treibt unseren Tourismus-ETF am stärksten?', pad=20)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# PLOT 2: Der Panik-Indikator (ETF vs. VIX / Marktvolatilität)
# ---------------------------------------------------------
print("Generiere Plot 2: ETF vs. VIX...")
fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.set_xlabel('Datum')
ax1.set_ylabel('EXV9 ETF Kurs (€)', color=color1)
ax1.plot(df_raw.index, df_raw['EXV9_ETF_Kurs'], color=color1, linewidth=2, label='Tourismus ETF')
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
vix_spalte = 'VIX_Kurs' if 'VIX_Kurs' in df_raw.columns else df_raw.columns[df_raw.columns.str.contains('VIX')][0]
ax2.set_ylabel('VIX (Markt-Volatilität / Panik)', color=color2)
ax2.plot(df_raw.index, df_raw[vix_spalte], color=color2, linewidth=1.5, alpha=0.7, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Gegenläufiger Trend: Tourismus-ETF vs. Markt-Volatilität (VIX)', pad=15)
fig.tight_layout()
plt.show()


# ---------------------------------------------------------
# PLOT 3: Saisonalität (ETF vs. Europäische Ferien-Welle)
# ---------------------------------------------------------
print("Generiere Plot 3: ETF vs. Ferien...")
# Dynamische Suche, falls 'vacation_summer_wave' fehlt
ferien_suchlauf = df_raw.columns[df_raw.columns.str.contains('vacation|ferien|holiday', case=False)]

if len(ferien_suchlauf) > 0:
    ferien_spalte = 'vacation_summer_wave' if 'vacation_summer_wave' in df_raw.columns else ferien_suchlauf[0]
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.set_xlabel('Datum')
    ax1.set_ylabel('EXV9 ETF Kurs (€)', color=color1)
    df_raw_ausschnitt = df_raw.loc['2023-01-01':'2024-12-31']
    ax1.plot(df_raw_ausschnitt.index, df_raw_ausschnitt['EXV9_ETF_Kurs'], color=color1, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Ferien-Intensität', color=color3)
    ax2.fill_between(df_raw_ausschnitt.index, df_raw_ausschnitt[ferien_spalte], color=color3, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor=color3)
    ax2.set_ylim(0, 1.2)

    plt.title('Saisonalität prüfen: Steigt der ETF zur Sommerferien-Saison? (Ausschnitt 23/24)', pad=15)
    fig.tight_layout()
    plt.show()
else:
    print("  -> Übersprungen: Keine Ferienspalte für Plot 3 gefunden.")


# ---------------------------------------------------------
# PLOT 4: Peer-Group Vergleich (Indexiert auf 100)
# ---------------------------------------------------------
print("Generiere Plot 4: Peer-Group Vergleich...")
plt.figure(figsize=(14, 7))

vergleichs_aktien = ['EXV9_ETF_Kurs', 'Lufthansa_Kurs', 'Booking_Kurs', 'Airbnb_Kurs']
vorhandene_aktien = [aktie for aktie in vergleichs_aktien if aktie in df_raw.columns]

df_raw_normiert = df_raw[vorhandene_aktien].copy().dropna()
if not df_raw_normiert.empty:
    df_raw_normiert = (df_raw_normiert / df_raw_normiert.iloc[0]) * 100

    for spalte in vorhandene_aktien:
        if 'ETF' in spalte:
            plt.plot(df_raw_normiert.index, df_raw_normiert[spalte], label='EXV9 ETF (Our Target)', linewidth=3, color='black')
        else:
            plt.plot(df_raw_normiert.index, df_raw_normiert[spalte], label=spalte.replace('_Kurs', ''), linewidth=1.5, alpha=0.7)

    plt.title('Performance-Vergleich: ETF vs. Einzelaktien (Start = 100 Punkte)', fontsize=16, pad=15)
    plt.ylabel('Performance (Indexiert)')
    plt.xlabel('Datum')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# PLOT 5: Kosten-Schock (Ölpreis vs. ETF)
# ---------------------------------------------------------
print("Generiere Plot 5: Ölpreis vs. ETF...")
fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.set_xlabel('Datum')
ax1.set_ylabel('EXV9 ETF Kurs (€)', color=color1)
ax1.plot(df_raw.index, df_raw['EXV9_ETF_Kurs'], color=color1, linewidth=2, label='ETF')
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = '#8c564b'
oel_spalte = 'BrentOil_Kurs' if 'BrentOil_Kurs' in df_raw.columns else df_raw.columns[df_raw.columns.str.contains('Oil')][0]

ax2.set_ylabel('Brent Rohöl Preis ($)', color=color2)
ax2.plot(df_raw.index, df_raw[oel_spalte], color=color2, linewidth=1.5, linestyle='-.', alpha=0.8)
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Makroökonomischer Einfluss: Tourismus-ETF vs. Rohölpreis (Kerosinkosten)', fontsize=16, pad=15)
fig.tight_layout()
plt.show()


# ---------------------------------------------------------
# PLOT 6: Gleitender Durchschnitt der News-Stimmung
# ---------------------------------------------------------
print("Generiere Plot 6: Rolling Sentiment...")
sentiment_suchlauf = df_raw.columns[df_raw.columns.str.contains('Sentiment|GDELT|news', case=False)]

if len(sentiment_suchlauf) > 0:
    sentiment_spalte = sentiment_suchlauf[0]
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.set_ylabel('EXV9 ETF Kurs (€)', color=color1)
    ax1.plot(df_raw.index, df_raw['EXV9_ETF_Kurs'], color=color1, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = '#2ca02c'
    geglättetes_sentiment = df_raw[sentiment_spalte].rolling(window=14).mean()

    ax2.set_ylabel('News Stimmung (14-Tage Durchschnitt)', color=color2)
    ax2.plot(df_raw.index, geglättetes_sentiment, color=color2, linewidth=2, label='14-Tage Sentiment-Trend')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('Der Stimmungs-Trend: 14-Tage gleitender Durchschnitt der News', fontsize=16, pad=15)
    fig.tight_layout()
    plt.show()
else:
    print("  -> Übersprungen: Keine News/Sentiment-Spalte für Plot 6 gefunden.")


# ---------------------------------------------------------
# PLOT 7: EXV9 5 Tage Return Verteilung
# ---------------------------------------------------------
print("Generiere Plot 7: Renditeverteilung...")
if "EXV9_ETF_Kurs_return_5d" in df_returns.columns:
    plt.figure(figsize=(10, 5))
    df_returns["EXV9_ETF_Kurs_return_5d"].hist(bins=40)
    plt.title("Verteilung der zukünftigen EXV9-Rendite (5 Tage)")
    plt.xlabel("EXV9_ETF_Kurs [%]")
    plt.ylabel("Anzahl")
    plt.show()


# ---------------------------------------------------------
# PLOT 8: Korrelationsmatrix Returns
# ---------------------------------------------------------
print("Generiere Plot 8: Korrelation Return Features...")
cols = ["EXV9_ETF_Kurs_return_1d", "DAX_Kurs_return_1d", "VIX_Kurs_return_1d", "BrentOil_Kurs_return_1d", "Lufthansa_Kurs_return_1d", "Booking_Kurs_return_1d"]
vorhandene_cols = [c for c in cols if c in df_returns.columns]

if len(vorhandene_cols) > 1:
    rename_map = {
        "EXV9_ETF_Kurs_return_1d": "EXV9", "DAX_Kurs_return_1d": "DAX",
        "VIX_Kurs_return_1d": "VIX", "BrentOil_Kurs_return_1d": "Brent",
        "Lufthansa_Kurs_return_1d": "LH", "Booking_Kurs_return_1d": "Booking"
    }
    plot_df = df_returns[vorhandene_cols].rename(columns=rename_map)
    plt.figure(figsize=(8,6))
    sns.heatmap(plot_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Korrelation der Return-Features")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# PLOT 9: Öl Plot & PLOT 10: Öl Plot Rolling
# ---------------------------------------------------------
if "BrentOil_Kurs_return_5d" in df_returns.columns and "EXV9_ETF_Kurs_return_5d" in df_returns.columns:
    print("Generiere Plot 9 & 10: Öl-Returns...")
    # Plot 9
    plt.figure(figsize=(12,5))
    plt.plot(df_returns.index, df_returns["BrentOil_Kurs_return_5d"], label="Brent")
    plt.plot(df_returns.index, df_returns["EXV9_ETF_Kurs_return_5d"], label="EXV9")
    plt.legend()
    plt.title("Brent vs EXV9")
    plt.show()

    # Plot 10
    plt.figure(figsize=(12,5))
    plt.plot(df_returns.index, df_returns["BrentOil_Kurs_return_5d"].rolling(30).mean(), label="Brent (30d MA)")
    plt.plot(df_returns.index, df_returns["EXV9_ETF_Kurs_return_5d"].rolling(30).mean(), label="EXV9 (30d MA)")
    plt.legend()
    plt.title("30-Tage Durchschnitt der 5-Tages-Renditen")
    plt.show()


# ---------------------------------------------------------
# PLOT: Ferien vs. geglättete ETF-Rendite
# ---------------------------------------------------------
if "EXV9_ETF_Kurs_return_5d" in df_returns.columns and "vacation_summer_wave" in df_raw.columns:
    print("Generiere Plot: Ferien vs. Geglättete Rendite...")
    fig, ax1 = plt.subplots(figsize=(14,6))
    etf_smooth = df_returns["EXV9_ETF_Kurs_return_5d"].rolling(30).mean()
    ax1.plot(df_returns.index, etf_smooth, linewidth=2, label="EXV9 Return (30 Tage geglättet)")
    ax1.set_ylabel("EXV9 Return [%]")

    ax2 = ax1.twinx()
    ax2.fill_between(df_raw.index, df_raw["vacation_summer_wave"], alpha=0.3)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Ferienintensität")
    plt.title("EXV9-Rendite und europäische Sommerferien")
    plt.show()


# ---------------------------------------------------------
# PLOT: Krankheitsinzidenz vs. ETF
# ---------------------------------------------------------
if "target_return_5d" in df_feature.columns and "ECDC_Europa_Inzidenz" in df_raw.columns:
    print("Generiere Plot: Krankheitsinzidenz vs. ETF...")
    df_feature["EXV9_return_5d_smooth"] = df_feature["target_return_5d"].rolling(window=20).mean()
    df_raw["ECDC_Europa_Inzidenz_smooth"] = df_raw["ECDC_Europa_Inzidenz"].rolling(window=20).mean()

    plt.figure(figsize=(14,6))
    ax1 = plt.gca()
    ax1.plot(df_feature.index, df_feature["EXV9_return_5d_smooth"], color="blue", linewidth=2, label="EXV9 Return (20d MA)")
    ax1.set_ylabel("EXV9 5D Return")

    ax2 = ax1.twinx()
    ax2.plot(df_raw.index, df_raw["ECDC_Europa_Inzidenz_smooth"], color="red", linewidth=2, alpha=0.8, label="Europa Inzidenz (20d MA)")
    ax2.set_ylabel("Europa Inzidenz")
    plt.title("Geglätteter EXV9-Return und Europa-Krankheitsinzidenz")
    plt.grid(alpha=0.3)
    plt.show()


# ---------------------------------------------------------
# Descriptive Stats & Ferien-Trendanalyse
# ---------------------------------------------------------
features = ["CentralEurope_Temp", "HolidayDestination_Temp", "Inflation_Gesamt", "target_return_5d", "target_trend_5d"]
vorhandene_feats = [f for f in features if f in df_feature.columns]
if vorhandene_feats:
    print("\nDeskriptive Statistik:")
    print(df_feature[vorhandene_feats].describe().round(2))

season_features = {"Sommer": "vacation_summer_wave", "Ostern": "vacation_easter_block", "Weihnachten": "vacation_christmas_block"}
vorhandene_seasons = {k: v for k, v in season_features.items() if v in df_feature.columns}

if len(vorhandene_seasons) == 3 and "target_trend_5d" in df_feature.columns:
    print("Generiere Plot: Positive Trends in Ferienperioden...")
    normal_days = np.ones(len(df_feature), dtype=bool)
    for feature in vorhandene_seasons.values():
        normal_days &= (df_feature[feature] == 0)

    normal_positive_trend = df_feature.loc[normal_days, "target_trend_5d"].mean() * 100
    results = []

    for season_name, feature in vorhandene_seasons.items():
        season_days = df_feature[feature] == 1
        positive_during_season = df_feature.loc[season_days, "target_trend_5d"].mean() * 100
        results.append({"Saison": season_name, "Ferienzeit": positive_during_season, "Normale Tage": normal_positive_trend})

    results_df = pd.DataFrame(results)
    x = np.arange(len(results_df))
    width = 0.35

    plt.figure(figsize=(11, 6))
    bars1 = plt.bar(x - width / 2, results_df["Ferienzeit"], width, label="Ferienzeit")
    bars2 = plt.bar(x + width / 2, results_df["Normale Tage"], width, label="Normale Tage")

    plt.ylabel("Anteil positiver Trends [%]")
    plt.xlabel("Ferienperiode")
    plt.title("Positive 5-Tages-Trends: Ferienzeit vs. normale Tage")
    plt.xticks(x, results_df["Saison"])
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{height:.1f}%", ha="center", va="bottom")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# PLOT: VIX Richtung vs Target Trend
# ---------------------------------------------------------
if "VIX_Kurs_return_5d" in df_feature.columns and "target_trend_5d" in df_feature.columns:
    print("Generiere Plot: Trendwahrscheinlichkeit nach VIX...")
    df_plot = df_feature.copy()
    df_plot["VIX_Richtung"] = df_plot["VIX_Kurs_return_5d"].apply(lambda x: "VIX steigt" if x > 0 else "VIX fällt")
    results = df_plot.groupby("VIX_Richtung")["target_trend_5d"].mean().mul(100).reset_index()
    results.columns = ["VIX_Richtung", "Positive Trends"]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(results["VIX_Richtung"], results["Positive Trends"])
    plt.ylabel("Anteil positiver EXV9-Trends [%]")
    plt.xlabel("VIX-Entwicklung der letzten 5 Tage")
    plt.title("EXV9-Trendwahrscheinlichkeit nach VIX-Entwicklung")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{height:.1f}%", ha="center", va="bottom")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# PLOT 11: Lead-Lag Analyse (Frühindikatoren identifizieren)
# ---------------------------------------------------------
if "EXV9_ETF_Kurs" in df_feature.columns:
    print("Generiere Plot 11: Lead-Lag Analyse...")
    returns = df_feature.select_dtypes(include=[np.number]).pct_change()
    lags = range(-5, 6)
    target = 'EXV9_ETF_Kurs'
    features_to_test = ['Ryanair_Kurs', 'VIX_Kurs', 'Booking_Kurs', 'BrentOil_Kurs']
    features_to_test = [f for f in features_to_test if f in returns.columns]

    if features_to_test:
        lag_results = []
        for f in features_to_test:
            for lag in lags:
                corr = returns[target].corr(returns[f].shift(lag))
                lag_results.append({'Feature': f.replace('_Kurs', ''), 'Lag (Tage)': lag, 'Korrelation': corr})

        df_lags = pd.DataFrame(lag_results)
        plt.figure(figsize=(12, 7))
        sns.lineplot(data=df_lags, x='Lag (Tage)', y='Korrelation', hue='Feature', marker='o', linewidth=2, markersize=8)
        plt.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.6)
        plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        plt.title('Lead-Lag Analyse: Wer bewegt sich ZUERST? (Frühindikatoren)', pad=20)
        plt.xlabel('Verzögerung in Tagen\n<-- Feature passiert NACH dem ETF (Nutzenlos)  |  Feature passiert VOR dem ETF (Frühindikator) -->')
        plt.ylabel('Korrelation mit heutiger ETF-Rendite')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------
# PLOT 12: Sektor-Divergenz (Verdeckte Dynamiken im ETF)
# ---------------------------------------------------------
if "EXV9_ETF_Kurs" in df_feature.columns:
    print("Generiere Plot 12: Sektor-Divergenz Analyse...")
    airlines = [c for c in ['Lufthansa_Kurs', 'Ryanair_Kurs', 'EasyJet_Kurs'] if c in df_feature.columns]
    hotels = [c for c in ['Booking_Kurs', 'Marriott_Hotels_Kurs', 'Airbnb_Kurs'] if c in df_feature.columns]

    if airlines and hotels:
        df_sectors = df_feature[['EXV9_ETF_Kurs']].copy()
        df_sectors['Airlines_Durchschnitt'] = df_feature[airlines].mean(axis=1)
        df_sectors['Hotels_Durchschnitt'] = df_feature[hotels].mean(axis=1)
        df_sectors = df_sectors.dropna()

        if not df_sectors.empty:
            df_sectors_norm = (df_sectors / df_sectors.iloc[0]) * 100
            plt.figure(figsize=(14, 7))
            plt.plot(df_sectors_norm.index, df_sectors_norm['EXV9_ETF_Kurs'], label='EXV9 ETF (Aggregiertes Ziel)', color='black', linewidth=4)
            plt.plot(df_sectors_norm.index, df_sectors_norm['Airlines_Durchschnitt'], label='Sektor: Airlines', color='#1f77b4', linestyle='-.', linewidth=2)
            plt.plot(df_sectors_norm.index, df_sectors_norm['Hotels_Durchschnitt'], label='Sektor: Hotels & Booking', color='#ff7f0e', linestyle='--', linewidth=2)
            plt.title('Verdeckte Dynamiken: Divergenz innerhalb des ETF-Korbs', pad=15)
            plt.ylabel('Performance (Indexiert = 100)')
            plt.xlabel('Datum')
            plt.legend()
            plt.tight_layout()
            plt.show()


# ---------------------------------------------------------
# PLOT: Inflation vs. Geglättete Rendite
# ---------------------------------------------------------
if "target_return_5d" in df_feature.columns and "Inflation_Gesamt" in df_feature.columns:
    print("Generiere Plot: Inflation vs. Rendite...")
    df_feature["ETF_smooth"] = df_feature["target_return_5d"].rolling(window=20).mean()
    df_feature["Inflation_smooth"] = df_feature["Inflation_Gesamt"].rolling(window=20).mean()

    fig, ax1 = plt.subplots(figsize=(14,6))
    ax1.plot(df_feature.index, df_feature["ETF_smooth"], color="blue", linewidth=2, label="EXV9 Return (20d MA)")
    ax1.set_ylabel("EXV9 5D Return", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(df_feature.index, df_feature["Inflation_smooth"], color="red", linewidth=2, alpha=0.8, label="Inflation Gesamt (20d MA)")
    ax2.set_ylabel("Inflationsindex", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title("Geglätteter EXV9-Return und Inflation Gesamt")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

print("Alle Plots erfolgreich generiert!")