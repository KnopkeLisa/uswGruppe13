import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

print("Lade KI-Wörterbuch herunter...")
# Das muss man nur einmalig herunterladen
nltk.download('vader_lexicon', quiet=True)

# 1. Die KI initialisieren
vader = SentimentIntensityAnalyzer()

# 2. Die rohen News-Daten laden
print("Lade Google News...")
df_news = pd.read_csv('data/google_news.csv', sep=';')

# Leere Zeilen in der Spalte 'title' abfangen, damit nichts abstürzt
df_news['title'] = df_news['title'].fillna("")

# 3. Datum standardisieren
df_news['date'] = pd.to_datetime(df_news['published'], errors='coerce').dt.normalize()

# 4. Die Magie: Text in Zahlen umwandeln (-1 bis 1)
print("Berechne Stimmungswerte (Sentiment)...")
df_news['News_Sentiment'] = df_news['title'].apply(
    lambda text: vader.polarity_scores(str(text))['compound']
)

# 5. Aufräumen: Wir behalten nur das saubere Datum und den errechneten Wert
df_clean = df_news[['date', 'News_Sentiment']].copy()

# Da es an einem Tag (z.B. am 2026-05-29) mehrere Artikel gibt,
# fassen wir diese zusammen und nehmen den Durchschnitt der Stimmung an diesem Tag.
df_clean = df_clean.groupby('date').mean().reset_index()

# 6. Speichern der reinen Zahlen-CSV für den finalen Merge
ausgabe = 'data/google_news_sentiment.csv'
# Wir speichern es jetzt als normales Komma-CSV ab, das macht den Merge einfacher
df_clean.to_csv(ausgabe, index=False, sep=',')

print(f"✅ Fertig! Die Schlagzeilen wurden in Zahlen übersetzt und unter '{ausgabe}' gespeichert.")