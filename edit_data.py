import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Design für die Plots hübsch und professionell machen
sns.set_theme(style="whitegrid", context="talk")

print("Lade finale Matrix für die Plots...")
# Wir laden unsere frisch gemergte Tabelle
df = pd.read_csv('finale_matrix.csv', index_col='Datum', parse_dates=True)

# ---------------------------------------------------------
# PLOT 1: Die Korrelations-Heatmap (Der absolute Prof-Liebling)
# Zeigt: Welche Variablen bewegen sich mit dem ETF mit?
# ---------------------------------------------------------
print("Generiere Plot 1: Korrelations-Heatmap...")
plt.figure(figsize=(10, 8))

# Wir berechnen die Korrelation aller Spalten MIT DEM ETF KURS
# und nehmen die Top 10 stärksten Zusammenhänge (positiv wie negativ)
korrelationen = df.corr()['EXV9_ETF_Kurs'].drop('EXV9_ETF_Kurs')
top_features = korrelationen.abs().sort_values(ascending=False).head(10).index
spalten_fuer_heatmap = ['EXV9_ETF_Kurs'] + list(top_features)

# Heatmap zeichnen
sns.heatmap(df[spalten_fuer_heatmap].corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Top 10 Zusammenhänge: Was treibt unseren Tourismus-ETF am stärksten?', pad=20)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# PLOT 2: Der Panik-Indikator (ETF vs. VIX / Marktvolatilität)
# Zeigt: Wenn die Angst im Markt steigt, fällt der Tourismus.
# ---------------------------------------------------------
print("Generiere Plot 2: ETF vs. VIX...")
fig, ax1 = plt.subplots(figsize=(14, 6))

color1 = '#1f77b4' # Blau
ax1.set_xlabel('Datum')
ax1.set_ylabel('EXV9 ETF Kurs (€)', color=color1)
ax1.plot(df.index, df['EXV9_ETF_Kurs'], color=color1, linewidth=2, label='Tourismus ETF')
ax1.tick_params(axis='y', labelcolor=color1)

# Zweite Y-Achse für den VIX (Angstbarometer)
ax2 = ax1.twinx()
color2 = '#d62728' # Rot
# Falls eure VIX-Spalte etwas anders heißt (z.B. VIX_Close), hier anpassen!
vix_spalte = 'VIX_Kurs' if 'VIX_Kurs' in df.columns else df.columns[df.columns.str.contains('VIX')][0]
ax2.set_ylabel('VIX (Markt-Volatilität / Panik)', color=color2)
ax2.plot(df.index, df[vix_spalte], color=color2, linewidth=1.5, alpha=0.7, linestyle='--')
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
df_ausschnitt = df.loc['2023-01-01':'2024-12-31']
ax1.plot(df_ausschnitt.index, df_ausschnitt['EXV9_ETF_Kurs'], color=color1, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color3 = '#ff7f0e' # Orange
# Suche die Spalte mit den Ferien (vacation_wave oder aehnlich)
ferien_spalte = 'vacation_summer_wave' if 'vacation_summer_wave' in df_ausschnitt.columns else df_ausschnitt.columns[df_ausschnitt.columns.str.contains('vacation|ferien', case=False)][0]
ax2.set_ylabel('Ferien-Intensität', color=color3)
ax2.fill_between(df_ausschnitt.index, df_ausschnitt[ferien_spalte], color=color3, alpha=0.3)
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
vorhandene_aktien = [aktie for aktie in vergleichs_aktien if aktie in df.columns]

# WICHTIG: Alle auf 100 normieren (Wert / Startwert * 100)
df_normiert = df[vorhandene_aktien].copy()
df_normiert = df_normiert.dropna() # Leere Starttage ausblenden
df_normiert = (df_normiert / df_normiert.iloc[0]) * 100

for spalte in vorhandene_aktien:
    if 'ETF' in spalte:
        plt.plot(df_normiert.index, df_normiert[spalte], label='EXV9 ETF (Unser Modell-Ziel)', linewidth=3, color='black')
    else:
        plt.plot(df_normiert.index, df_normiert[spalte], label=spalte.replace('_Kurs', ''), linewidth=1.5, alpha=0.7)

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

color1 = '#1f77b4'
ax1.set_xlabel('Datum')
ax1.set_ylabel('EXV9 ETF Kurs (€)', color=color1)
ax1.plot(df.index, df['EXV9_ETF_Kurs'], color=color1, linewidth=2, label='ETF')
ax1.tick_params(axis='y', labelcolor=color1)

# Zweite Y-Achse für das Öl
ax2 = ax1.twinx()
color2 = '#8c564b' # Braun für Öl
oel_spalte = 'BrentOil_Kurs' if 'BrentOil_Kurs' in df.columns else df.columns[df.columns.str.contains('Oil')][0]

ax2.set_ylabel('Brent Rohöl Preis ($)', color=color2)
ax2.plot(df.index, df[oel_spalte], color=color2, linewidth=1.5, linestyle='-.', alpha=0.8)
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
sentiment_spalte = 'News_Sentiment' if 'News_Sentiment' in df.columns else df.columns[df.columns.str.contains('Sentiment|GDELT')][0]

fig, ax1 = plt.subplots(figsize=(14, 6))

# Linke Achse: ETF
color1 = '#1f77b4'
ax1.set_ylabel('EXV9 ETF Kurs (€)', color=color1)
ax1.plot(df.index, df['EXV9_ETF_Kurs'], color=color1, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)

# Rechte Achse: Sentiment mit "Rolling Mean" (Glättung)
ax2 = ax1.twinx()
color2 = '#2ca02c' # Grün
# Hier passiert die Magie: Wir glätten die Stimmung über 14 Tage!
geglättetes_sentiment = df[sentiment_spalte].rolling(window=14).mean()

ax2.set_ylabel('News Stimmung (14-Tage Durchschnitt)', color=color2)
ax2.plot(df.index, geglättetes_sentiment, color=color2, linewidth=2, label='14-Tage Sentiment-Trend')
# Nulllinie einzeichnen (Grenze zwischen Positiven und Negativen News)
ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Der Stimmungs-Trend: 14-Tage gleitender Durchschnitt der News', fontsize=16, pad=15)
fig.tight_layout()
plt.show()

print("✅ Alle Plots erfolgreich generiert!")