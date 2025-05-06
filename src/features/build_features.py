"""
build_features.py
Applies TF-IDF vectorization.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

def get_vectorizer():
    return TfidfVectorizer(max_features=5000)

def vectorize_text(vectorizer, text_data):
    return vectorizer.fit_transform(text_data)
