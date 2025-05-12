from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

# Load the cleaned data
df_combined = pd.read_csv('cleaned_data.csv')

# Remove any rows with missing text data
df_combined.dropna(subset=['text'], inplace=True)

# Convert text data into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df_combined['text'])
y = df_combined['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the split data and vectorizer
joblib.dump((X_train, X_test, y_train, y_test), 'train_test_split_data.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
