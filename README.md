# Vorhersage des europäischen Tourismussektors durch kombinierte Datenströme

## 1. Projektthema & Zielsetzung
Dieses Projekt entwickelt ein KI-Modell (ein neuronales Netz), um die kurzfristige Kursentwicklung der europäischen Reise- und Freizeitbranche vorherzusagen. 

Im Kern untersuchen wir, ob eine KI durch die mathematische Verknüpfung von **klassischen europäischen Finanzmarktdaten** mit **lokalisierten, alternativen Datenquellen** (Wetter-Clustern, Google-Suchanfragen, Gesundheitsdaten und Medien-Stimmungen) einen Informationsvorsprung am Aktienmarkt erzielen kann. Der Fokus auf den europäischen Kontinent ermöglicht es uns, regionale Saisonalitäten, Ferienwellen und Wettereffekte präzise zu modellieren.

*   **Untersuchungszeitraum:** 01.01.2023 bis 30.05.2026
*   **Vorhersage-Zeitraum:** Exakt 5 Handelstage in die Zukunft (Trendprognose für eine Handelswoche).
*   **Daten-Frequenz:** Täglich (1d)

---

## 2. Die Zielvariable (Das Target $y$)
Um das Risiko von unvorhersehbaren Ereignissen bei einzelnen Unternehmen zu minimieren, prognostiziert das Modell den führenden europäischen Branchen-Index:

*   **Zielvariable ($y$):** `EXV9.DE` (STOXX Europe 600 Travel & Leisure UCITS ETF)
*   **Bedeutung:** Dieser Index beinhaltet die kapitalstärksten Unternehmen Europas aus den Segmenten Luftfahrt, Hotellerie, Gastronomie und Unterhaltung. Er bildet die aggregierte wirtschaftliche Entwicklung des europäischen Tourismussektors ab.

---

## 3. Eingabevariablen (Features $X$)
Die Feature-Matrix ist durch das bestehende Datenbeschaffungs-Skript fest definiert und setzt sich aus den folgenden Dateien zusammen (siehe Struktur in 1.PNG und 2.PNG):

### A. Finanzmarktdaten & Branchen-Segmente (`yfinance`)
*   **Europäische Fluggesellschaften:** `LHA.DE` (Lufthansa), `RYAAY` (Ryanair), `AF.PA` (Air France-KLM) und `EZJ.L` (EasyJet) zur Abbildung des kontinentalen Linien- und Billigflugmarktes.
*   **Digitale Buchungsplattformen & Tech:** `BKNG` (Booking Holdings), `ABNB` (Airbnb) und `EXPE` (Expedia) sowie der `AWAY` ETF (Global Travel Tech ETF).
*   **Hotellerie & Mobilität:** `HLT` (Hilton Hotels), `MAR` (Marriott Hotels) und `UBER` (Uber) zur Erfassung von Unterkunft- und Transporttrends.
*   **Branchen-Benchmarks (ETFs):** `JETS` (U.S. Global Jets ETF) und `PEJ` (Invesco Leisure and Entertainment ETF) als sektorübergreifende Kontrollvariablen.
*   **Marktumfeld & Volatilität:** `^GDAXI` (DAX) als europäischer Leitindex und `^VIX` (CBOE Volatility Index) als Gradmesser für die allgemeine Nervosität am globalen Aktienmarkt.

### B. Wirtschaftliche Treiber & Währungen
*   **Energiekosten-Hebel:** `BZ=F` (Brent Crude Oil) und `CL=F` (WTI Crude Oil) zur Messung des direkten Kostendrucks durch Kerosinpreise.
*   **Währungs-Hebel:** `EURUSD=X` (Euro/US-Dollar), `EURTRY=X` (Euro/Türkische Lira), `EURGBP=X` (Euro/Britisches Pfund) und `EURCHF=X` (Euro/Schweizer Franken), um die Attivitätsunterschiede europäischer Destinationen zu erfassen.
*   **Makro-Inflation (`inflation.csv`):** Offizielle europäische Verbraucherpreisindizes aus der Eurostat-Datenbank (`prc_hicp_midx`). 
    *   *Frequenz-Korrektur:* Da offizielle Inflationsdaten nur monatlich und zeitverzögert erscheinen, führt die Pipeline ein mathematisches **Upsampling auf tägliche Basis** (mittels linearer Interpolation und einem eingebauten Time-Lag von 15 Tagen) durch. Dies verhindert, dass das Modell mit Daten aus der Zukunft trainiert wird (*Data Leakage*).

### C. Alternative Daten & Saisonalitäten
*   **Wetterdaten (`wetter_openmeteo.csv`):** Täglich aggregierte Werte (Durchschnittstemperatur, Niederschlagssumme und Sonnenscheindauer) über die Open-Meteo-API für vier strategische europäische Knotenpunkte und Urlaubsregionen: **Berlin, Palma (Mallorca), Antalya und Rom**.
*   **Digitale Nachfrage (`google_trends.csv`):** Suchvolumen für die Begriffe: `"Mallorca"`, `"wellness"`, `"airbnb"`, `"hotel"` und `"camping"`. (Wöchentliche Daten werden über einen *Forward-Fill* an die tägliche Aktienfrequenz angepasst).
*   **Europäische Kalender-Effekte (`german_holidays_daily.csv`):** Um der europäischen Zielvariable gerecht zu werden, wurde die ursprüngliche rein deutsche Betrachtung im Code auf ein **aggregiertes europäisches Feiertags-Feature** (`holidays.EuropeanUnion`) angehoben. Es liefert einen gewichteten Aktivitäts-Index für Handels- und Reisetage in den europäischen Kernmärkten.
*   **Gesundheitsdaten (`ARE-Inzidenz.csv` / `GrippeDaten.csv`):** Zeitreihen des Robert-Koch-Instituts (RKI) zu akuten Atemwegserkrankungen (ARE) und Influenza-Werten, um den Einfluss lokaler Krankheitswellen als Stichprobe für das mitteleuropäische Reiseverhalten zu testen.
*   **Europäische Gesundheitsdaten (`ecdc_respiratory_data.csv`):** Die offiziellen Überwachungsdaten des **ECDC (European Centre for Disease Prevention and Control)** für akute Atemwegserkrankungen (Influenza und RSV) auf EU-Ebene. Da diese Daten wöchentlich erhoben werden, erfolgt in der Pipeline ein tägliches Upsampling (Interpolation), um den Einfluss kontinentaler Krankheitswellen auf die kurzfristige Reisenachfrage fehlerfrei zu messen.
*   **Nachrichten-Sentiment (`google_news.csv`):** Täglich gecrawlte Schlagzeilen über den Google-News-RSS-Feed, basierend auf den risiko- und nachfragespezifischen Suchbegriffen: `"airport strike"`, `"flight cancellation"`, `"travel warning"`, `"tourism"` und `"hotel booking"`.