import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Design für die Plots hübsch und professionell machen
sns.set_theme(style="whitegrid", context="talk")
# Wir laden unsere frisch gemergte Tabelle
df_raw = pd.read_csv('./data-prep-pre-split/raw-data-matrix.csv', index_col='Datum', parse_dates=True)
df_returns = pd.read_csv('./data-prep-pre-split/returns_matrix.csv', index_col='Datum', parse_dates=True)

color1 = '#1f77b4' # Blau
color2 = '#d62728' # Rot
color3 = '#ff7f0e' # Orange

# # ---------------------------------------------------------
# # PLOT 1: Die Korrelations-Heatmap (Der absolute Prof-Liebling)
# # Zeigt: Welche Variablen bewegen sich mit dem ETF mit?
# # ---------------------------------------------------------
# print("Generiere Plot 1: Korrelations-Heatmap...")
# plt.figure(figsize=(10, 8))

# # Wir berechnen die Korrelation aller Spalten MIT DEM ETF KURS
# # und nehmen die Top 10 stärksten Zusammenhänge (positiv wie negativ)
# korrelationen = df_raw.corr()['EXV9_ETF_Kurs'].drop('EXV9_ETF_Kurs')
# top_features = korrelationen.abs().sort_values(ascending=False).head(10).index
# spalten_fuer_heatmap = ['EXV9_ETF_Kurs'] + list(top_features)

# # Heatmap zeichnen
# sns.heatmap(df_raw[spalten_fuer_heatmap].corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
# plt.title('Top 10 Zusammenhänge: Was treibt unseren Tourismus-ETF am stärksten?', pad=20)
# plt.tight_layout()
# plt.show()


# # ---------------------------------------------------------
# # PLOT 2: Der Panik-Indikator (ETF vs. VIX / Marktvolatilität)
# # Zeigt: Wenn die Angst im Markt steigt, fällt der Tourismus.
# # ---------------------------------------------------------
# print("Generiere Plot 2: ETF vs. VIX...")
# fig, ax1 = plt.subplots(figsize=(14, 6))


# ax1.set_xlabel('Datum')
# ax1.set_ylabel('EXV9 ETF Kurs (€)', color=color1)
# ax1.plot(df_raw.index, df_raw['EXV9_ETF_Kurs'], color=color1, linewidth=2, label='Tourismus ETF')
# ax1.tick_params(axis='y', labelcolor=color1)

# # Zweite Y-Achse für den VIX (Angstbarometer)
# ax2 = ax1.twinx()
# # Falls eure VIX-Spalte etwas anders heißt (z.B. VIX_Close), hier anpassen!
# vix_spalte = 'VIX_Kurs' if 'VIX_Kurs' in df_raw.columns else df_raw.columns[df_raw.columns.str.contains('VIX')][0]
# ax2.set_ylabel('VIX (Markt-Volatilität / Panik)', color=color2)
# ax2.plot(df_raw.index, df_raw[vix_spalte], color=color2, linewidth=1.5, alpha=0.7, linestyle='--')
# ax2.tick_params(axis='y', labelcolor=color2)

# plt.title('Gegenläufiger Trend: Tourismus-ETF vs. Markt-Volatilität (VIX)', pad=15)
# fig.tight_layout()
# plt.show()


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

# # ---------------------------------------------------------
# # PLOT 4: Peer-Group Vergleich (Indexiert auf 100)
# # Zeigt: Ist der ETF stabiler als riskante Einzelaktien?
# # ---------------------------------------------------------
# print("Generiere Plot 4: Peer-Group Vergleich...")
# plt.figure(figsize=(14, 7))

# # Wir wählen den ETF und 3 bekannte Tourismus-Player (Namen ggf. anpassen!)
# vergleichs_aktien = ['EXV9_ETF_Kurs', 'Lufthansa_Kurs', 'Booking_Kurs', 'Airbnb_Kurs']

# # Wir prüfen, welche davon wirklich in eurer Matrix existieren
# vorhandene_aktien = [aktie for aktie in vergleichs_aktien if aktie in df_raw.columns]

# # WICHTIG: Alle auf 100 normieren (Wert / Startwert * 100)
# df_raw_normiert = df_raw[vorhandene_aktien].copy()
# df_raw_normiert = df_raw_normiert.dropna() # Leere Starttage ausblenden
# df_raw_normiert = (df_raw_normiert / df_raw_normiert.iloc[0]) * 100

# for spalte in vorhandene_aktien:
#     if 'ETF' in spalte:
#         plt.plot(df_raw_normiert.index, df_raw_normiert[spalte], label='EXV9 ETF (Unser Modell-Ziel)', linewidth=3, color='black')
#     else:
#         plt.plot(df_raw_normiert.index, df_raw_normiert[spalte], label=spalte.replace('_Kurs', ''), linewidth=1.5, alpha=0.7)

# plt.title('Performance-Vergleich: ETF vs. Einzelaktien (Start = 100 Punkte)', fontsize=16, pad=15)
# plt.ylabel('Performance (Indexiert)')
# plt.xlabel('Datum')
# plt.legend()
# plt.tight_layout()
# plt.show()

# # ---------------------------------------------------------
# # PLOT 5: Kosten-Schock (Ölpreis vs. ETF)
# # Zeigt: Drücken hohe Energiekosten die Kurse der Tourismusbranche?
# # ---------------------------------------------------------
# print("Generiere Plot 5: Ölpreis vs. ETF...")
# fig, ax1 = plt.subplots(figsize=(14, 6))

# ax1.set_xlabel('Datum')
# ax1.set_ylabel('EXV9 ETF Kurs (€)', color=color1)
# ax1.plot(df_raw.index, df_raw['EXV9_ETF_Kurs'], color=color1, linewidth=2, label='ETF')
# ax1.tick_params(axis='y', labelcolor=color1)

# # Zweite Y-Achse für das Öl
# ax2 = ax1.twinx()
# color2 = '#8c564b' # Braun für Öl
# oel_spalte = 'BrentOil_Kurs' if 'BrentOil_Kurs' in df_raw.columns else df_raw.columns[df_raw.columns.str.contains('Oil')][0]

# ax2.set_ylabel('Brent Rohöl Preis ($)', color=color2)
# ax2.plot(df_raw.index, df_raw[oel_spalte], color=color2, linewidth=1.5, linestyle='-.', alpha=0.8)
# ax2.tick_params(axis='y', labelcolor=color2)

# plt.title('Makroökonomischer Einfluss: Tourismus-ETF vs. Rohölpreis (Kerosinkosten)', fontsize=16, pad=15)
# fig.tight_layout()
# plt.show()

# # ---------------------------------------------------------
# # PLOT 6: Gleitender Durchschnitt der News-Stimmung
# # Zeigt: Wie ein Stimmungs-Trend den Kurs vorausahnen kann
# # ---------------------------------------------------------
# print("Generiere Plot 6: Rolling Sentiment...")

# # Wir prüfen, ob ihr das Sentiment oder GDELT nutzt
# sentiment_spalte = 'News_Sentiment' if 'News_Sentiment' in df_raw.columns else df_raw.columns[df_raw.columns.str.contains('Sentiment|GDELT')][0]

# fig, ax1 = plt.subplots(figsize=(14, 6))

# # Linke Achse: ETF
# ax1.set_ylabel('EXV9 ETF Kurs (€)', color=color1)
# ax1.plot(df_raw.index, df_raw['EXV9_ETF_Kurs'], color=color1, linewidth=2)
# ax1.tick_params(axis='y', labelcolor=color1)

# # Rechte Achse: Sentiment mit "Rolling Mean" (Glättung)
# ax2 = ax1.twinx()
# color2 = '#2ca02c' # Grün
# # Hier passiert die Magie: Wir glätten die Stimmung über 14 Tage!
# geglättetes_sentiment = df_raw[sentiment_spalte].rolling(window=14).mean()

# ax2.set_ylabel('News Stimmung (14-Tage Durchschnitt)', color=color2)
# ax2.plot(df_raw.index, geglättetes_sentiment, color=color2, linewidth=2, label='14-Tage Sentiment-Trend')
# # Nulllinie einzeichnen (Grenze zwischen Positiven und Negativen News)
# ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
# ax2.tick_params(axis='y', labelcolor=color2)

# plt.title('Der Stimmungs-Trend: 14-Tage gleitender Durchschnitt der News', fontsize=16, pad=15)
# fig.tight_layout()
# plt.show()

# print("Alle Plots erfolgreich generiert!")

# # ---------------------------------------------------------
# # PLOT 7: EXV9 5 Tage Return
# # Zeigt: Wie verhält sich der 5 Tage Return der Ziel Variable
# # ---------------------------------------------------------

# df_returns["EXV9_ETF_Kurs_return_5d"].hist(bins=40)

# plt.title("Verteilung der zukünftigen EXV9-Rendite (5 Tage)")
# plt.xlabel("EXV9_ETF_Kurs [%]")
# plt.ylabel("Anzahl")
# plt.show()

# #---------------------------------------------------------
# # PLOT 8: Korrelationsmatrix
# #---------------------------------------------------------

# cols = [
#     "EXV9_ETF_Kurs_return_1d",
#     "DAX_Kurs_return_1d",
#     "VIX_Kurs_return_1d",
#     "BrentOil_Kurs_return_1d",
#     "Lufthansa_Kurs_return_1d",
#     "Booking_Kurs_return_1d",
# ]

# rename_map = {
#     "EXV9_ETF_Kurs_return_1d": "EXV9",
#     "DAX_Kurs_return_1d": "DAX",
#     "VIX_Kurs_return_1d": "VIX",
#     "BrentOil_Kurs_return_1d": "Brent",
#     "Lufthansa_Kurs_return_1d": "LH",
#     "Booking_Kurs_return_1d": "Booking"
# }

# plot_df = df_returns[cols].rename(columns=rename_map)

# plt.figure(figsize=(8,6))

# sns.heatmap(
#     plot_df.corr(),
#     annot=True,
#     fmt=".2f",
#     cmap="coolwarm",
#     center=0,
#     square=True
# )

# plt.title("Korrelation der Return-Features")
# plt.tight_layout()
# plt.show()

# #---------------------------------------------------------
# # PLOT 9: Öl Plot
# #---------------------------------------------------------

# plt.figure(figsize=(12,5))

# plt.plot(
#     df_returns.index,
#     df_returns["BrentOil_Kurs_return_5d"],
#     label="Brent"
# )

# plt.plot(
#     df_returns.index,
#     df_returns["EXV9_ETF_Kurs_return_5d"],
#     label="EXV9"
# )

# plt.legend()
# plt.title("Brent vs EXV9")
# plt.show()

# #---------------------------------------------------------
# # PLOT 10: Öl Plot - Rolling
# #---------------------------------------------------------

# plt.figure(figsize=(12,5))

# plt.plot(
#     df_returns.index,
#     df_returns["BrentOil_Kurs_return_5d"]
#         .rolling(30)
#         .mean(),
#     label="Brent (30d MA)"
# )

# plt.plot(
#     df_returns.index,
#     df_returns["EXV9_ETF_Kurs_return_5d"]
#         .rolling(30)
#         .mean(),
#     label="EXV9 (30d MA)"
# )

# plt.legend()
# plt.title("30-Tage Durchschnitt der 5-Tages-Renditen")
# plt.show()


# ---------------------------------------------------------
# PLOT: Ferien vs. Nicht-Ferien
# Zeigt: Unterscheiden sich die Renditen?
# ---------------------------------------------------------

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