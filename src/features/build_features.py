from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Apply TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df_combined['text'])
y = df_combined['label']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
