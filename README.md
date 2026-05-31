#Vorhersage des globalen Tourismussektors durch kombinierte Datenströme

#1. Projektthema & Zielsetzung
Dieses Projekt entwickelt ein KI-Modell (ein neuronales Netz), um die kurzfristige Kursentwicklung der weltweiten Reise- und Freizeitbranche vorherzusagen. 

Im Kern wollen wir herausfinden, ob eine KI durch die geschickte Verknüpfung von **klassischen Finanzmarktdaten** (wie Aktien- und Ölpreisen) mit **alternativen Datenquellen** (wie Wetterdaten, Google-Suchanfragen und Nachrichten-Stimmungen) einen Informationsvorsprung am Aktienmarkt erzielen kann.

*   **Vorhersage-Zeitraum:** 5 Handelstage in die Zukunft (Trendprognose für eine Woche).
*   **Methodik:** Zeitreihen-Analyse unter Berücksichtigung von saisonalen Mustern und politischen Ereignissen.

---

# 2. Die Zielvariable (Was wir vorhersagen)
Um das Risiko von unvorhersehbaren Ereignissen bei einzelnen Unternehmen (wie z. B. ein plötzlicher Chef-Wechsel oder ein lokaler Skandal) zu minimieren, sagen wir keinen Einzelwert voraus, sondern einen großen Branchen-Index:

*   **Zielvariable:** `EXV1.DE` (STOXX Global 1800 Travel & Leisure ETF)
*   **Bedeutung:** Dieser ETF bündelt die größten und erfolgreichsten Konzerne der Welt aus den Bereichen Luftfahrt, Hotels, Buchungsplattformen und Kreuzfahrten. Er ist das perfekte Abbild für die wirtschaftliche Gesundheit der gesamten Tourismusbranche.

---

#3. Eingabevariablen (Features)
Unsere Datenmatrix setzt sich aus verschiedenen Bausteinen zusammen, die sich gegenseitig ergänzen:

# A. Branchen- & Segment-Daten
*   **Fluggesellschaften:** Der `JETS` ETF (U.S. Global Jets), um die weltweiten Trends und Kapazitäten im Flugverkehr zu erfassen.
*   **Digitale Buchungsplattformen:** Die Aktienkurse von Marktführern wie `BKNG` (Booking Holdings), `ABNB` (Airbnb), `EXPE` (Expedia Group) und `TRIP` (TripAdvisor).
*   **Massentourismus:** Die `TUI1.DE` Aktie (TUI AG) als Gradmesser für das klassische, preisbewusste Pauschalreise-Geschäft.
*   **Luxustourismus:** Der `GLUX.PA` ETF (Amundi S&P Global Luxury), um die Entwicklung im Premium-Segment zu beobachten, da wohlhabende Kunden oft krisenresistenter sind.

#B. Wirtschaftliche Treiber (Kosten- & Währungsfaktoren)
*   **Sprit- & Energiekosten:** Die Rohölpreise `BZ=F` (Brent) und `CL=F` (WTI). Höhere Ölpreise bedeuten meist direkt höhere Kosten für Airlines und somit teurere Tickets.
*   **US-Dollar-Stärke:** Der `DX-Y.NYB` (US-Dollar-Index), da Wechselkurse einen riesigen Einfluss darauf haben, wie attraktiv Fernreisen für Urlauber sind.

#C. Alternative Daten (Reiselust & Stimmung)
*   **Wetterdaten:** Täglich zusammengefasste Temperaturen aus wichtigen weltweiten Tourismus-Regionen (wie Mallorca, Orlando, Bangkok, Cancun, Kapstadt, Sydney), um kurzfristige, wetterbedingte Buchungswellen zu erkennen.
*   **Google Trends (Die reale Nachfrage):** Das aktuelle Suchvolumen für die Begriffe `"book flight"` (Flug buchen) und `"book hotel"` (Hotel buchen) als direkter Frühindikator.
*   **Nachrichten-Stimmung (Sentiment):** Ein automatisch berechneter Wert aus weltweiten Wirtschaftsmeldungen, der zeigt, ob gerade negative Themen (wie Zoll-Androhungen oder Streiks) oder positive Nachrichten (wie Visa-Erleichterungen) die Medien dominieren.

---

#4. Diskussion: Warum wir keine Krankheitsdaten (RKI) nutzen
Wir haben im Team intensiv geprüft, ob wir Infektionsdaten (z. B. RKI-Daten zu Atemwegserkrankungen) einbauen sollten. Wir haben uns aus zwei logischen Gründen **dagegen** entschieden:
1.  **Geografischer Widerspruch:** Die RKI-Daten bilden nur die Situation in Deutschland ab. Da unsere Zielvariable (`EXV1.DE`) aber ein weltweiter Index ist, macht es mathematisch keinen Sinn, ihn mit rein lokalen Krankheitswellen vorhersagen zu wollen.
2.  **Fehlende Datenqualität:** Nach der akuten Corona-Phase werden internationale Infektionsdaten nicht mehr in der nötigen täglichen und standardisierten Form über Schnittstellen (APIs) bereitgestellt, die für unser Modell notwendig wäre.