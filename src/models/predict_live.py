import tweepy
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved Random Forest model and the vectorizer
best_rf = joblib.load('best_rf_model.pkl')  # Load the best Random Forest model
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = joblib.load(f)  # Load the TF-IDF vectorizer

# Twitter API setup
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAMvqzwEAAAAACgV%2BJd5ciPeOj8GZcWDPHGPC0%2Bk%3D4355VvXuONwjRFg1o30OzLs0SzPLZHJBb3tJbtRdW6Ut6RVHIx'
client = tweepy.Client(bearer_token=bearer_token)

# Fetching recent tweets mentioning "fake news"
query = "fake news"
response = client.search_recent_tweets(query=query, tweet_fields=["author_id", "created_at"], max_results=10)

# Cleaning tweet text
def clean_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"http\S+|www\S+", '', tweet)  # Remove URLs
    tweet = re.sub(r"@\w+|#\w+", '', tweet)  # Remove mentions and hashtags
    tweet = re.sub(r"[^a-zA-Z\s]", '', tweet)  # Remove non-alphabetic characters
    return tweet

# Extracting and cleaning tweet texts
if response.data:
    cleaned_tweets = [clean_tweet(tweet.text) for tweet in response.data]
    X_live = vectorizer.transform(cleaned_tweets)  # Transform live tweets using the saved vectorizer
    
    # Make predictions using the best Random Forest model
    preds = best_rf.predict(X_live)
    
    print("Live Tweet Predictions:\n")
    for i, tweet in enumerate(response.data):
        label = "Real" if preds[i] == 1 else "Fake"
        print(f"Tweet: {tweet.text[:120]}...\n→ Predicted Label: {label}\n")
else:
    print("No tweets found with the given query.")
