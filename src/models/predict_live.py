"""
predict_live.py
Fetches and classifies live tweets using trained model and vectorizer.
"""

import tweepy

def fetch_tweets(bearer_token, query="fake news", count=10):
    client = tweepy.Client(bearer_token=bearer_token)
    response = client.search_recent_tweets(query=query, tweet_fields=["author_id", "created_at"], max_results=count)
    return response.data if response.data else []
