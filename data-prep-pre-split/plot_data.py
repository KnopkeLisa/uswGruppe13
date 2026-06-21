import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# # Design für die Plots hübsch und professionell machen
sns.set_theme(style="whitegrid", context="talk")
# # Wir laden unsere frisch gemergte Tabelle
df_raw = pd.read_csv('./data-prep-pre-split/raw-data-matrix.csv', index_col='Datum', parse_dates=True)
df_returns = pd.read_csv('./data-prep-pre-split/returns_matrix.csv', index_col='Datum', parse_dates=True)
df_feature = pd.read_csv('./data-prep-pre-split/feature_matrix.csv', index_col='Datum', parse_dates=True)

color1 = '#1f77b4' # Blau
color2 = '#d62728' # Rot
color3 = '#ff7f0e' # Orange

# ---------------------------------------------------------
# PLOT 1: Die Korrelations-Heatmap (Der absolute Prof-Liebling)
# Zeigt: Welche Variablen bewegen sich mit dem ETF mit?
# ---------------------------------------------------------
print("Generiere Plot 1: Korrelations-Heatmap...")
plt.figure(figsize=(10, 8))

# Wir berechnen die Korrelation aller Spalten MIT DEM ETF KURS
# und nehmen die Top 10 stärksten Zusammenhänge (positiv wie negativ)
korrelationen = df_raw.corr()['EXV9_ETF_Kurs'].drop('EXV9_ETF_Kurs')
top_features = korrelationen.abs().sort_values(ascending=False).head(10).index
spalten_fuer_heatmap = ['EXV9_ETF_Kurs'] + list(top_features)

# Heatmap zeichnen
sns.heatmap(df_raw[spalten_fuer_heatmap].corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Top 10 Zusammenhänge: Was treibt unseren Tourismus-ETF am stärksten?', pad=20)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# PLOT 2: Der Panik-Indikator (ETF vs. VIX / Marktvolatilität)
# Zeigt: Wenn die Angst im Markt steigt, fällt der Tourismus.
# ---------------------------------------------------------
print("Generiere Plot 2: ETF vs. VIX...")
fig, ax1 = plt.subplots(figsize=(14, 6))


ax1.set_xlabel('Datum')
ax1.set_ylabel('EXV9 ETF Kurs (€)', color=color1)
ax1.plot(df_raw.index, df_raw['EXV9_ETF_Kurs'], color=color1, linewidth=2, label='Tourismus ETF')
ax1.tick_params(axis='y', labelcolor=color1)

# Zweite Y-Achse für den VIX (Angstbarometer)
ax2 = ax1.twinx()
# Falls eure VIX-Spalte etwas anders heißt (z.B. VIX_Close), hier anpassen!
vix_spalte = 'VIX_Kurs' if 'VIX_Kurs' in df_raw.columns else df_raw.columns[df_raw.columns.str.contains('VIX')][0]
ax2.set_ylabel('VIX (Markt-Volatilität / Panik)', color=color2)
ax2.plot(df_raw.index, df_raw[vix_spalte], color=color2, linewidth=1.5, alpha=0.7, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Gegenläufiger Trend: Tourismus-ETF vs. Markt-Volatilität (VIX)', pad=15)
fig.tight_layout()
plt.show()


# ---------------------------------------------------------
# PLOT 3: Saisonalität (ETF vs. Europäische Ferien-Welle)
# Zeigt: Sehen wir den klassischen "Sommer-Boost" in den Kursen?
# ---------------------------------------------------------
print("Generiere Plot 3: ETF vs. Ferien...")
fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.set_xlabel('Datum')
ax1.set_ylabel('EXV9 ETF Kurs (€)', color=color1)
# Wir plotten mal nur die Jahre 2023 und 2024, damit man die Wellen besser sieht
df_raw_ausschnitt = df_raw.loc['2023-01-01':'2024-12-31']
ax1.plot(df_raw_ausschnitt.index, df_raw_ausschnitt['EXV9_ETF_Kurs'], color=color1, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()

# Suche die Spalte mit den Ferien (vacation_wave oder aehnlich)
ferien_spalte = 'vacation_summer_wave' if 'vacation_summer_wave' in df_raw_ausschnitt.columns else df_raw_ausschnitt.columns[df_raw_ausschnitt.columns.str.contains('vacation|ferien', case=False)][0]
ax2.set_ylabel('Ferien-Intensität', color=color3)
ax2.fill_between(df_raw_ausschnitt.index, df_raw_ausschnitt[ferien_spalte], color=color3, alpha=0.3)
ax2.tick_params(axis='y', labelcolor=color3)
ax2.set_ylim(0, 1.2) # Skala von 0 bis 1

plt.title('Saisonalität prüfen: Steigt der ETF zur Sommerferien-Saison? (Ausschnitt 23/24)', pad=15)
fig.tight_layout()
plt.show()

# ---------------------------------------------------------
# PLOT 4: Peer-Group Vergleich (Indexiert auf 100)
# Zeigt: Ist der ETF stabiler als riskante Einzelaktien?
# ---------------------------------------------------------
print("Generiere Plot 4: Peer-Group Vergleich...")
plt.figure(figsize=(14, 7))

# Wir wählen den ETF und 3 bekannte Tourismus-Player (Namen ggf. anpassen!)
vergleichs_aktien = ['EXV9_ETF_Kurs', 'Lufthansa_Kurs', 'Booking_Kurs', 'Airbnb_Kurs']

# Wir prüfen, welche davon wirklich in eurer Matrix existieren
vorhandene_aktien = [aktie for aktie in vergleichs_aktien if aktie in df_raw.columns]

# WICHTIG: Alle auf 100 normieren (Wert / Startwert * 100)
df_raw_normiert = df_raw[vorhandene_aktien].copy()
df_raw_normiert = df_raw_normiert.dropna() # Leere Starttage ausblenden
df_raw_normiert = (df_raw_normiert / df_raw_normiert.iloc[0]) * 100

for spalte in vorhandene_aktien:
    if 'ETF' in spalte:
        plt.plot(df_raw_normiert.index, df_raw_normiert[spalte], label='EXV9 ETF (Unser Modell-Ziel)', linewidth=3, color='black')
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
# Zeigt: Drücken hohe Energiekosten die Kurse der Tourismusbranche?
# ---------------------------------------------------------
print("Generiere Plot 5: Ölpreis vs. ETF...")
fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.set_xlabel('Datum')
ax1.set_ylabel('EXV9 ETF Kurs (€)', color=color1)
ax1.plot(df_raw.index, df_raw['EXV9_ETF_Kurs'], color=color1, linewidth=2, label='ETF')
ax1.tick_params(axis='y', labelcolor=color1)

# Zweite Y-Achse für das Öl
ax2 = ax1.twinx()
color2 = '#8c564b' # Braun für Öl
oel_spalte = 'BrentOil_Kurs' if 'BrentOil_Kurs' in df_raw.columns else df_raw.columns[df_raw.columns.str.contains('Oil')][0]

ax2.set_ylabel('Brent Rohöl Preis ($)', color=color2)
ax2.plot(df_raw.index, df_raw[oel_spalte], color=color2, linewidth=1.5, linestyle='-.', alpha=0.8)
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Makroökonomischer Einfluss: Tourismus-ETF vs. Rohölpreis (Kerosinkosten)', fontsize=16, pad=15)
fig.tight_layout()
plt.show()

# ---------------------------------------------------------
# PLOT 6: Gleitender Durchschnitt der News-Stimmung
# Zeigt: Wie ein Stimmungs-Trend den Kurs vorausahnen kann
# ---------------------------------------------------------
print("Generiere Plot 6: Rolling Sentiment...")

# Wir prüfen, ob ihr das Sentiment oder GDELT nutzt
sentiment_spalte = 'News_Sentiment' if 'News_Sentiment' in df_raw.columns else df_raw.columns[df_raw.columns.str.contains('Sentiment|GDELT')][0]

fig, ax1 = plt.subplots(figsize=(14, 6))

# Linke Achse: ETF
ax1.set_ylabel('EXV9 ETF Kurs (€)', color=color1)
ax1.plot(df_raw.index, df_raw['EXV9_ETF_Kurs'], color=color1, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)

# Rechte Achse: Sentiment mit "Rolling Mean" (Glättung)
ax2 = ax1.twinx()
color2 = '#2ca02c' # Grün
# Hier passiert die Magie: Wir glätten die Stimmung über 14 Tage!
geglättetes_sentiment = df_raw[sentiment_spalte].rolling(window=14).mean()

ax2.set_ylabel('News Stimmung (14-Tage Durchschnitt)', color=color2)
ax2.plot(df_raw.index, geglättetes_sentiment, color=color2, linewidth=2, label='14-Tage Sentiment-Trend')
# Nulllinie einzeichnen (Grenze zwischen Positiven und Negativen News)
ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Der Stimmungs-Trend: 14-Tage gleitender Durchschnitt der News', fontsize=16, pad=15)
fig.tight_layout()
plt.show()

print("Alle Plots erfolgreich generiert!")

# ---------------------------------------------------------
# PLOT 7: EXV9 5 Tage Return
# Zeigt: Wie verhält sich der 5 Tage Return der Ziel Variable
# ---------------------------------------------------------

df_returns["EXV9_ETF_Kurs_return_5d"].hist(bins=40)

plt.title("Verteilung der zukünftigen EXV9-Rendite (5 Tage)")
plt.xlabel("EXV9_ETF_Kurs [%]")
plt.ylabel("Anzahl")
plt.show()

#---------------------------------------------------------
# PLOT 8: Korrelationsmatrix
#---------------------------------------------------------

cols = [
    "EXV9_ETF_Kurs_return_1d",
    "DAX_Kurs_return_1d",
    "VIX_Kurs_return_1d",
    "BrentOil_Kurs_return_1d",
    "Lufthansa_Kurs_return_1d",
    "Booking_Kurs_return_1d",
]

rename_map = {
    "EXV9_ETF_Kurs_return_1d": "EXV9",
    "DAX_Kurs_return_1d": "DAX",
    "VIX_Kurs_return_1d": "VIX",
    "BrentOil_Kurs_return_1d": "Brent",
    "Lufthansa_Kurs_return_1d": "LH",
    "Booking_Kurs_return_1d": "Booking"
}

plot_df = df_returns[cols].rename(columns=rename_map)

plt.figure(figsize=(8,6))

sns.heatmap(
    plot_df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True
)

plt.title("Korrelation der Return-Features")
plt.tight_layout()
plt.show()

#---------------------------------------------------------
# PLOT 9: Öl Plot
#---------------------------------------------------------

plt.figure(figsize=(12,5))

plt.plot(
    df_returns.index,
    df_returns["BrentOil_Kurs_return_5d"],
    label="Brent"
)

plt.plot(
    df_returns.index,
    df_returns["EXV9_ETF_Kurs_return_5d"],
    label="EXV9"
)

plt.legend()
plt.title("Brent vs EXV9")
plt.show()

#---------------------------------------------------------
# PLOT 10: Öl Plot - Rolling
#---------------------------------------------------------

plt.figure(figsize=(12,5))

plt.plot(
    df_returns.index,
    df_returns["BrentOil_Kurs_return_5d"]
        .rolling(30)
        .mean(),
    label="Brent (30d MA)"
)

plt.plot(
    df_returns.index,
    df_returns["EXV9_ETF_Kurs_return_5d"]
        .rolling(30)
        .mean(),
    label="EXV9 (30d MA)"
)

plt.legend()
plt.title("30-Tage Durchschnitt der 5-Tages-Renditen")
plt.show()


# ---------------------------------------------------------
# PLOT: Ferien vs. geglättete ETF-Rendite
# ---------------------------------------------------------

fig, ax1 = plt.subplots(figsize=(14,6))

etf_smooth = (
    df_returns["EXV9_ETF_Kurs_return_5d"]
    .rolling(30)
    .mean()
)

ax1.plot(
    df_returns.index,
    etf_smooth,
    linewidth=2,
    label="EXV9 Return (30 Tage geglättet)"
)

ax1.set_ylabel("EXV9 Return [%]")

ax2 = ax1.twinx()

ax2.fill_between(
    df_raw.index,
    df_raw["vacation_summer_wave"],
    alpha=0.3
)

ax2.set_ylim(0, 1.1)
ax2.set_ylabel("Ferienintensität")

plt.title(
    "EXV9-Rendite und europäische Sommerferien"
)

plt.show()

df_feature["EXV9_return_5d_smooth"] = (
    df_feature["target_return_5d"]
    .rolling(window=20)
    .mean()
)

df_raw["ECDC_Europa_Inzidenz_smooth"] = (
    df_raw["ECDC_Europa_Inzidenz"]
    .rolling(window=20)
    .mean()
)

plt.figure(figsize=(14,6))

ax1 = plt.gca()

ax1.plot(
    df_feature.index,
    df_feature["EXV9_return_5d_smooth"],
    color="blue",
    linewidth=2,
    label="EXV9 Return (20d MA)"
)

ax1.set_ylabel("EXV9 5D Return")

ax2 = ax1.twinx()

ax2.plot(
    df_raw.index,
    df_raw["ECDC_Europa_Inzidenz_smooth"],
    color="red",
    linewidth=2,
    alpha=0.8,
    label="Europa Inzidenz (20d MA)"
)

ax2.set_ylabel("Europa Inzidenz")

plt.title(
    "Geglätteter EXV9-Return und Europa-Krankheitsinzidenz"
)

plt.grid(alpha=0.3)

plt.show()


features = [
    "CentralEurope_Temp",
    "HolidayDestination_Temp",
    "Inflation_Gesamt",
    "target_return_5d",
    "target_trend_5d"
]

print(
    df_feature[features]
    .describe()
    .round(2)
)

# ---------------------------------------------------------
# PLOT 7: Lead-Lag Analyse (Frühindikatoren identifizieren)
# Zeigt: Welche Variablen bewegen sich VOR dem ETF?
# ---------------------------------------------------------
print("Generiere Plot 7: Lead-Lag Analyse...")

# Wir brauchen Renditen für saubere Korrelationen, nicht absolute Preise
returns = df_feature.pct_change()

lags = range(-5, 6)
target = 'EXV9_ETF_Kurs'
features_to_test = ['Ryanair_Kurs', 'VIX_Kurs', 'Booking_Kurs', 'BrentOil_Kurs']
features_to_test = [f for f in features_to_test if f in returns.columns]

lag_results = []
for f in features_to_test:
    for lag in lags:
        # shift() vergleicht den heutigen ETF mit dem zeitversetzten Feature
        corr = returns[target].corr(returns[f].shift(lag))
        lag_results.append({'Feature': f.replace('_Kurs', ''), 'Lag (Tage)': lag, 'Korrelation': corr})

df_lags = pd.DataFrame(lag_results)

plt.figure(figsize=(12, 7))
sns.lineplot(data=df_lags, x='Lag (Tage)', y='Korrelation', hue='Feature', marker='o', linewidth=2, markersize=8)

# Optische Hilfslinien zur besseren Lesbarkeit
plt.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.6)
plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)

plt.title('Lead-Lag Analyse: Wer bewegt sich ZUERST? (Frühindikatoren)', pad=20)
plt.xlabel('Verzögerung in Tagen\n<-- Feature passiert NACH dem ETF (Nutzenlos)  |  Feature passiert VOR dem ETF (Frühindikator) -->')
plt.ylabel('Korrelation mit heutiger ETF-Rendite')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# PLOT 8: Sektor-Divergenz (Verdeckte Dynamiken im ETF)
# Zeigt: Wie Airlines und Hotels sich intern aufspalten
# ---------------------------------------------------------
print("Generiere Plot 8: Sektor-Divergenz Analyse...")

# Branchen logisch gruppieren
airlines = [c for c in ['Lufthansa_Kurs', 'Ryanair_Kurs', 'EasyJet_Kurs'] if c in df_feature.columns]
hotels = [c for c in ['Booking_Kurs', 'Marriott_Hotels_Kurs', 'Airbnb_Kurs'] if c in df_feature.columns]

df_sectors = df_feature[['EXV9_ETF_Kurs']].copy()
df_sectors['Airlines_Durchschnitt'] = df_feature[airlines].mean(axis=1)
df_sectors['Hotels_Durchschnitt'] = df_feature[hotels].mean(axis=1)

# NaN-Werte droppen, damit der Startpunkt für die Normierung sauber ist
df_sectors = df_sectors.dropna()

# Alle Werte auf 100 normieren (Wert / Erster Wert * 100)
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

plt.figure(figsize=(14,6))

# ETF glätten
df_feature["ETF_smooth"] = (
    df_feature["target_return_5d"]
    .rolling(window=20)
    .mean()
)

# Inflation glätten
df_feature["Inflation_smooth"] = (
    df_feature["Inflation_Gesamt"]
    .rolling(window=20)
    .mean()
)

ax1 = plt.gca()

# ETF
ax1.plot(
    df_feature.index,
    df_feature["ETF_smooth"],
    color="blue",
    linewidth=2,
    label="EXV9 Return (20d MA)"
)

ax1.set_ylabel("EXV9 5D Return", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

# Zweite Achse
ax2 = ax1.twinx()

ax2.plot(
    df_feature.index,
    df_feature["Inflation_smooth"],
    color="red",
    linewidth=2,
    alpha=0.8,
    label="Inflation Gesamt (20d MA)"
)

ax2.set_ylabel("Inflationsindex", color="red")
ax2.tick_params(axis="y", labelcolor="red")

plt.title(
    "Geglätteter EXV9-Return und Inflation Gesamt"
)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

season_features = {
    "Sommer": "vacation_summer_wave",
    "Ostern": "vacation_easter_block",
    "Weihnachten": "vacation_christmas_block"
}

#echte Normalzeit: keine der betrachteten Ferienperioden aktiv
normal_days = np.ones(len(df_feature), dtype=bool)

for feature in season_features.values():
    normal_days &= (df_feature[feature] == 0)

normal_positive_trend = (
    df_feature.loc[
        normal_days,
        "target_trend_5d"
    ].mean() * 100
)

print("Normale Tage:", normal_days.sum())

results = []

for season_name, feature in season_features.items():

    season_days = df_feature[feature] == 1

    positive_during_season = (
        df_feature.loc[
            season_days,
            "target_trend_5d"
        ].mean() * 100
    )

    print(feature, season_days.sum())

    results.append({
        "Saison": season_name,
        "Ferienzeit": positive_during_season,
        "Normale Tage": normal_positive_trend
    })

results_df = pd.DataFrame(results)

x = np.arange(len(results_df))
width = 0.35

plt.figure(figsize=(11, 6))

bars1 = plt.bar(
    x - width / 2,
    results_df["Ferienzeit"],
    width,
    label="Ferienzeit"
)

bars2 = plt.bar(
    x + width / 2,
    results_df["Normale Tage"],
    width,
    label="Normale Tage"
)

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

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.1f}%",
            ha="center",
            va="bottom"
        )

plt.tight_layout()
plt.show()

df_plot = df_feature.copy()

# VIX positiv = Angst steigt
# VIX negativ = Angst sinkt

df_plot["VIX_Richtung"] = df_plot["VIX_Kurs_return_5d"].apply(
    lambda x: "VIX steigt" if x > 0 else "VIX fällt"
)

results = (
    df_plot
    .groupby("VIX_Richtung")["target_trend_5d"]
    .mean()
    .mul(100)
    .reset_index()
)

results.columns = ["VIX_Richtung", "Positive Trends"]

plt.figure(figsize=(8, 6))

bars = plt.bar(
    results["VIX_Richtung"],
    results["Positive Trends"]
)

plt.ylabel("Anteil positiver EXV9-Trends [%]")
plt.xlabel("VIX-Entwicklung der letzten 5 Tage")
plt.title("EXV9-Trendwahrscheinlichkeit nach VIX-Entwicklung")

plt.ylim(0, 100)
plt.grid(axis="y", alpha=0.3)

for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.1f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()