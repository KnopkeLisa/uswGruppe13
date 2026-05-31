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
    "PEJ": "PEJ",
    "Uber":"UBER",
    "BrentOil": "BZ=F",
    "WTIOil": "CL=F",
    "EUR_USD": "EURUSD=X",
    "EUR_TRY": "EURTRY=X",
    "EUR_GPB": "EURGBP=X",
    "EUR_CHF": "EURCHF=X",
    "DAX": "^GDAXI",
    "VIX": "^VIX",
    "AWAY-ETF": "AWAY"
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
# Feiertage
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

# ============================================================
# Inflation
# ============================================================

def download_inflation():
    """
    Lädt Inflationsdaten ueber eurostat
    und speichert diese als CSV.
    """
    print("\n=== Inflation Data Download ===")
    df = eurostat.get_data_df("prc_hicp_midx")
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
        start_date="2025-01-01",
        end_date="2025-03-31"
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

    timeline.to_csv(filename, index=False)

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
    pd.concat(rows).to_csv(
        os.path.join(OUTPUT_DIR,"wetter_openmeteo.csv"),
        index=False
    )

# ============================================================
# Gesundheitsdaten
# ============================================================

def downloadGesundheitsdaten():
    urls = {
        "grippeweb": "https://raw.githubusercontent.com/robert-koch-institut/GrippeWeb_Daten_des_Wochenberichts/main/GrippeWeb_Daten_des_Wochenberichts.tsv",
        "are_konsultationsinzidenz": "https://raw.githubusercontent.com/robert-koch-institut/ARE-Konsultationsinzidenz/main/ARE-Konsultationsinzidenz.tsv"
    }

    for name, url in urls.items():
            df = pd.read_csv(url, sep="\t")
            df = df[df["Kalenderwoche"] >= "2023-W01"]
            df.to_csv(
            os.path.join(OUTPUT_DIR, f"{name}.csv"),
            index=False
        )

    print(f"{name} gespeichert: {len(df)} Zeilen")


def download_europe_health_data():
    """
    Lädt die wöchentlichen Inzidenzraten akuter Atemwegserkrankungen (ILIARIRates).
    """
    print("\n=== ECDC Europe Health Data Download ===")

    raw_url = "https://raw.githubusercontent.com/EU-ECDC/Respiratory_viruses_weekly_data/main/data/ILIARIRates.csv"
    filename = os.path.join(OUTPUT_DIR, "ECDC_Europa_Inzidenz.csv")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    print("Lade ILIARIRates.csv von ECDC GitHub-Server...")
    response = requests.get(raw_url, headers=headers, timeout=60)

    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Erfolgreich gespeichert unter: {filename}")
    else:
        print(f"❌ Fehler beim Download von ECDC: Status Code {response.status_code}")
# ============================================================
# Main
# ============================================================

def main():

    print("Starte Datenbeschaffung!")

    #download_stock_data()
    #download_google_trends()
    #download_holidays()
    #download_inflation()
    #download_gdelt_news()
    #download_google_rss_news()
    #downloadWetter()
    downloadGesundheitsdaten()
    #download_europe_health_data()


    print(f"\nDaten gespeichert unter:\n{OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
