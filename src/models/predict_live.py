import tweepy
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved Random Forest model and vectorizer
best_rf = joblib.load('best_rf_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Twitter API setup (you need a valid API token)
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAMvqzwEAAAAACgV%2BJd5ciPeOj8GZcWDPHGPC0%2Bk%3D4355VvXuONwjRFg1o30OzLs0SzPLZHJBb3tJbtRdW6Ut6RVHIx'
client = tweepy.Client(bearer_token=bearer_token)

# Fetch recent tweets mentioning "fake news"
query = "fake news"
response = client.search_recent_tweets(query=query, tweet_fields=["author_id", "created_at"], max_results=10)

# Clean the tweet text
def clean_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"http\S+|www\S+", '', tweet)
    tweet = re.sub(r"@\w+|#\w+", '', tweet)
    tweet = re.sub(r"[^a-zA-Z\s]", '', tweet)
    return tweet

# If we found any tweets, make predictions
if response.data:
    cleaned_tweets = [clean_tweet(tweet.text) for tweet in response.data]
    X_live = vectorizer.transform(cleaned_tweets)

    # Predict using the saved Random Forest model
    preds = best_rf.predict(X_live)

    # Display the results
    for i, tweet in enumerate(response.data):
        label = "Real" if preds[i] == 1 else "Fake"
        print(f"Tweet: {tweet.text[:120]}...\n→ Predicted: {label}")
else:
    print("No tweets found.")
