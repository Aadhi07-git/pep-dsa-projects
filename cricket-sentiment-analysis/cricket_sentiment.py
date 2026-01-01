import pandas as pd

print("VS Code setup successful")

# Load Kaggle cricket tweets dataset
df = pd.read_csv("data/cricket_tweets.csv")

print("Dataset Loaded ✅")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nSample Tweets:")
print(df.head())
# STEP 3: Sentiment Analysis using TextBlob

from textblob import TextBlob

def get_sentiment(text):
    if pd.isna(text):
        return "neutral"
    analysis = TextBlob(str(text))
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity < 0:
        return "negative"
    else:
        return "neutral"

# Apply sentiment analysis on tweet text column
df["sentiment"] = df["text"].apply(get_sentiment)

print("\nSentiment column added ✅")
print(df[["text", "sentiment"]].head())
print("\nSentiment counts:")
print(df['sentiment'].value_counts())
df.to_csv("data/cricket_with_sentiment.csv", index=False)
print("File saved as cricket_with_sentiment.csv ✅")
import pandas as pd
import matplotlib.pyplot as plt
# Visualization: Sentiment Distribution
sentiment_counts = df['sentiment'].value_counts()

plt.figure()
sentiment_counts.plot(kind='bar')
plt.title('Sentiment Distribution of Cricket Tweets')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.savefig("sentiment_bar_chart.png", dpi=300, bbox_inches="tight")
plt.show()
# ---------- PIE CHART: Sentiment Distribution ----------

sentiment_counts = df['sentiment'].value_counts()

plt.figure()
plt.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct='%1.1f%%',
    startangle=90
)
plt.title('Sentiment Share of Cricket Tweets')
plt.savefig("sentiment_pie_chart.png", dpi=300, bbox_inches="tight")
plt.show()
plt.figure()

df.boxplot(
    column='retweets',
    by='sentiment'
)

plt.title('Retweets Distribution by Sentiment')
plt.suptitle('')   # removes automatic pandas title
plt.xlabel('Sentiment')
plt.ylabel('Number of Retweets')

plt.savefig("retweet_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()

