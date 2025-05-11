# Twitter API fetch tweets
import tweepy

client = tweepy.Client(bearer_token="YOUR_TWITTER_API_TOKEN")
query = 'fake news'
tweets = client.search_recent_tweets(query=query, max_results=100)

# Clean and transform tweets
tweet_texts = [clean_text(tweet.text) for tweet in tweets.data]
X_live = tfidf_vectorizer.transform(tweet_texts)

# Predict using the best model (e.g., Logistic Regression)
live_predictions = best_lr.predict(X_live)
print(live_predictions)
