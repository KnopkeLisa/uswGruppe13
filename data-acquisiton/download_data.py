import os
import pandas as pd
import yfinance as yf
import holidays
import eurostat
import requests
import feedparser
from pytrends.request import TrendReq
from gdeltdoc import GdeltDoc, Filters
from pathlib import Path

# ============================================================
# Konfiguration
# ============================================================

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"

TICKERS = {
    "EXV9_ETF": "EXV9.DE", #Zielvariable
    "AWAY-ETF": "AWAY",
    "Lufthansa": "LHA.DE",
    "Ryanair": "RYAAY",
    "Air France-KLM": "AF.PA",
    "EasyJet": "EZJ.L",
    "Booking": "BKNG",
    "Airbnb": "ABNB",
    "Expedia": "EXPE",
    "Hilton_Hotels": "HLT",
    "Marriott_Hotels":"MAR",
    "JETS": "JETS",
    "Uber":"UBER",
    "PrimeEnergyOil_Gas": "PEJ",
    "BrentOil": "BZ=F",
    "WTIOil": "CL=F",
    "EUR_USD": "EURUSD=X",
    "EUR_TRY": "EURTRY=X",
    "EUR_GPB": "EURGBP=X",
    "EUR_CHF": "EURCHF=X",
    "DAX": "^GDAXI",
    "VIX": "^VIX"
    
}

START_DATE = "2023-01-01"
END_DATE = "2026-05-30"
INTERVAL = "1d"


# ============================================================
# Yahoo Finance Daten
# ============================================================

def download_stock_data():
    """
    Lädt historische Finanzdaten von Yahoo Finance
    und speichert Daten als CSV.
    """

    print("\n=== Yahoo Finance Download ===\n")

    for name, ticker in TICKERS.items():
        print(f"Lade {name} ({ticker})...")
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            interval=INTERVAL,
            auto_adjust=True,
            progress=False
        )
        filename = os.path.join(
            OUTPUT_DIR,
            f"{name}.csv"
        )
        df.to_csv(filename)


# ============================================================
# Google Trends
# ============================================================

def download_google_trends():
    """
    Lädt Google Trends Daten
    und speichert diese als CSV.
    """

    print("\n=== Google Trends Download ===")

    pytrends = TrendReq(hl="de-DE")
    pytrends.build_payload(
        ["Mallorca", "wellness", "airbnb","hotel","camping"],
        timeframe=f"{START_DATE} {END_DATE}"
    )
    df = pytrends.interest_over_time()
    df = df.drop(columns=["isPartial"])

    filename = os.path.join(
        OUTPUT_DIR,
        "google_trends.csv"
    )
    df.to_csv(filename)

# ============================================================
# Feiertage und Schulferien
# ============================================================

def download_holidays():
    """
    Lädt Feiertage inkl. 0/1 Spalte
    und speichert diese als CSV.
    """
    print("\n=== Holidays Download ===")

    de_holidays = holidays.Germany(
        years=[2023, 2024, 2025, 2026]
    )
    # Alle Tage erzeugen
    daily = pd.DataFrame({
        "date": pd.date_range(
            start=START_DATE,
            end=END_DATE,
            freq="D"
        )
    })
    # Standard = kein Feiertag
    daily["holiday"] = 0
    # Feiertage markieren
    holiday_dates = pd.to_datetime(
        list(de_holidays.keys())
    )
    daily.loc[
        daily["date"].isin(holiday_dates),
        "holiday"
    ] = 1
    filename = os.path.join(
        OUTPUT_DIR,
        "german_holidays_daily.csv"
    )
    daily.to_csv(
        filename,
        index=False
    )

    def download_european_school_vacations():
        """
        Generiert ein tägliches Raster basierend auf START_DATE und END_DATE
        und teilt die europäischen Ferien in feste Blöcke (0/1) und fließende
        Wellen (Prozentwerte) auf.
        """
        print("\n=== Generating European School Vacation Indicators ===")

        daily = pd.DataFrame({
            "date": pd.date_range(start=START_DATE, end=END_DATE, freq="D")
        })

        daily["month"] = daily["date"].dt.month
        daily["day"] = daily["date"].dt.day
        daily["year"] = daily["date"].dt.year

        daily["vacation_christmas_block"] = 0.0
        daily["vacation_easter_block"] = 0.0
        daily["vacation_summer_wave"] = 0.0
        daily["vacation_autumn_wave"] = 0.0
        daily["vacation_winter_ski_wave"] = 0.0


        # DIE FESTEN BLOCK-FERIEN (Wert 0.0 oder 1.0)

        christmas_condition = (
                ((daily["month"] == 12) & (daily["day"] >= 21)) |
                ((daily["month"] == 1) & (daily["day"] <= 7))
        )
        daily.loc[christmas_condition, "vacation_christmas_block"] = 1.0

        easter_conditions = (
                ((daily["year"] == 2023) & (daily["month"] == 4) & (daily["day"].between(1, 16))) |  # Ostern: 09.04.23
                ((daily["year"] == 2024) & (daily["month"] == 3) & (daily["day"] >= 24)) |  # Ostern: 31.03.24
                ((daily["year"] == 2024) & (daily["month"] == 4) & (daily["day"] <= 7)) |
                ((daily["year"] == 2025) & (daily["month"] == 4) & (daily["day"].between(13, 27))) |  # Ostern: 20.04.25
                ((daily["year"] == 2026) & (daily["month"] == 3) & (daily["day"] >= 29)) |  # Ostern: 05.04.26
                ((daily["year"] == 2026) & (daily["month"] == 4) & (daily["day"] <= 12))
        )
        daily.loc[easter_conditions, "vacation_easter_block"] = 1.0

        # DIE FLIESSENDEN WELLEN-FERIEN (Prozentwerte)

        # Sommerferien-Welle
        daily.loc[daily["month"] == 6, "vacation_summer_wave"] = 0.3
        # Juli & August: Absolute Kernzeit, ganz Europa hat Ferien -> Peak
        daily.loc[daily["month"].isin([7, 8]), "vacation_summer_wave"] = 1.0
        # September: Spätzünder & letzte Staffeln -> Welle flacht ab
        daily.loc[daily["month"] == 9, "vacation_summer_wave"] = 0.2

        # Herbstferien-Welle
        daily.loc[daily["month"] == 10, "vacation_autumn_wave"] = 0.6
        daily.loc[((daily["month"] == 11) & (daily["day"] <= 3)), "vacation_autumn_wave"] = 0.2

        #Winter-/Skiferien-Welle
        daily.loc[daily["month"] == 2, "vacation_winter_ski_wave"] = 0.5

        daily = daily.drop(columns=["month", "day", "year"])

        filename = os.path.join(OUTPUT_DIR, "european_vacations_daily.csv")
        daily.to_csv(filename, index=False)
        print(f"✓ Schulferien-Matrix erfolgreich gespeichert unter: {filename}")

# ============================================================
# Inflation
# ============================================================

def download_inflation():
    """
    Lädt Inflationsdaten ueber eurostat
    und speichert diese als CSV.
    """
    print("\n=== Inflation Data Download ===")

    wanted_coicop = [
    "CP00",  # Gesamtinflation
    "CP01",  # Lebensmittel
    "CP04",  # Wohnen, Energie
    "CP07",  # Verkehr
    "CP09",  # Freizeit und Kultur
    "CP11",  # Gaststätten & Beherbergung
    "CP12"   # Sonstige Dienstleistungen
]
    df = eurostat.get_data_df("prc_hicp_midx")

    df = df[
    (df["geo\\TIME_PERIOD"] == "EA20") & #20 Euro Länder
    (df["coicop"].isin(wanted_coicop))
]
    df = pd.concat([
    df.iloc[:, :4], # Die ersten 4 Spalten bleiben
    df.iloc[:, 328:] # Zeiträume 1992-2022 fliegen raus
    ], axis=1)

    df.to_csv(
    os.path.join(OUTPUT_DIR, "inflation.csv"),
    index=False)

# ============================================================
# Weltweite News
# ============================================================   

def download_gdelt_news():

    """
    Lädt tägliches Nachrichten zu Tourismus-/Airline-Themen
    über GDELT und speichert es als CSV.
    """

    print("\n=== GDELT News Download ===\n")

    gd = GdeltDoc()

    filters = Filters(
        keyword="Lufthansa",
        start_date="2026-04-01",
        end_date="2026-05-01"
    )

    timeline = gd.timeline_search(
        "timelinevolraw",
        filters
    )
    timeline["datetime"] = pd.to_datetime(timeline["datetime"]).dt.date

    filename = os.path.join(
        OUTPUT_DIR,
        "gdelt_tourism_news.csv"
    )
    timeline.to_csv(filename, mode="a", header=not os.path.exists(filename), index=False)

def download_google_rss_news():
    print("\n=== Google RSS Download ===")
    SEARCH_TERMS = [
        "airport strike",
        "flight cancellation",
        "travel warning",
        "tourism",
        "hotel booking",
    ]

    rows = []

    for term in SEARCH_TERMS:

        feed = feedparser.parse(
            f"https://news.google.com/rss/search?q={term.replace(' ', '+')}"
        )

        for entry in feed.entries:

            rows.append({
                "published": pd.to_datetime(entry.published).date(),
                "keyword": term,
                "title": entry.title,
                "link": entry.link
            })
            
    print(f"Insgesamt gespeichert: {len(rows)}")
    df = pd.DataFrame(rows)

    csv_path = os.path.join(
        OUTPUT_DIR,
        "google_news.csv"
    )
    df.to_csv(
        csv_path,
        index=False,
        sep=";"
    )


# ============================================================
# Wetter
# ============================================================

def downloadWetter():

    print("\n=== Weather Data Download ===")
    # Open-Meteo historische Wetterdaten; Beispiel: Berlin, Palma, Antalya, Rom
    cities = {
        "Berlin": (52.52, 13.41),
        "Palma": (39.57, 2.65),
        "Antalya": (36.90, 30.71),
        "Rom": (41.90, 12.50),
    }
    rows = []
    for city, (lat, lon) in cities.items():
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": START_DATE, "end_date": END_DATE,
            "daily": "temperature_2m_mean,precipitation_sum,sunshine_duration",
            "timezone": "Europe/Berlin"
        }
        js = requests.get(url, params=params, timeout=60).json()["daily"]
        df = pd.DataFrame(js)
        df["city"] = city
        rows.append(df)
        print(f"{city} heruntergeladen: {len(df)} Zeilen")
    pd.concat(rows).to_csv(
        os.path.join(OUTPUT_DIR,"wetter_openmeteo.csv"),
        index=False
    )

# ============================================================
# Gesundheitsdaten
# ============================================================


def download_europe_health_data():

    print("\n=== Health Data Download ===")
    urls = {
    "grippeweb": {
        "url": "https://raw.githubusercontent.com/robert-koch-institut/GrippeWeb_Daten_des_Wochenberichts/main/GrippeWeb_Daten_des_Wochenberichts.tsv",
        "sep": "\t"
    },
    "are_konsultationsinzidenz": {
        "url": "https://raw.githubusercontent.com/robert-koch-institut/ARE-Konsultationsinzidenz/main/ARE-Konsultationsinzidenz.tsv",
        "sep": "\t"
    },
    "ECDC_Europa_Inzidenz": {
        "url": "https://raw.githubusercontent.com/EU-ECDC/Respiratory_viruses_weekly_data/main/data/ILIARIRates.csv",
        "sep": ","
    }
}
    for name, info in urls.items():
        df = pd.read_csv(info["url"], sep=info["sep"])
        if "Kalenderwoche" in df.columns:
            kw_col = "Kalenderwoche"
        elif "yearweek" in df.columns:
                kw_col = "yearweek"
        df = df[df[kw_col] >= "2023-W01"]
        df.to_csv(
            os.path.join(OUTPUT_DIR, f"{name}.csv"),
            index=False
    )
        print(f"{name} heruntergeladen: {len(df)} Zeilen")
# ============================================================
# Main
# ============================================================

def main():

    print("Starte Datenbeschaffung!")

    #download_stock_data()
    #download_google_trends()
    #download_google_rss_news()
    #download_holidays()
    #download_european_school_vacations()
    #download_inflation()
    #downloadWetter()
    #download_europe_health_data()

    #download_gdelt_news() #--> Führt des öfteren zu Fehlern, funktioniert aber

    print(f"\nDaten gespeichert unter:\n{OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
