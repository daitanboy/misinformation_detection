from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the data from the previous step (train/test split)
X_train, X_test, y_train, y_test = joblib.load('train_test_split_data.pkl')

# Model 1: Logistic Regression
param_grid_lr = {'C': [0.01, 0.1, 1, 10]}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, cv=5, scoring='f1')
grid_lr.fit(X_train, y_train)
best_lr = grid_lr.best_estimator_
print("Best Logistic Regression Model:", best_lr)

# Model 2: Random Forest
param_grid_rf = {'n_estimators': [100], 'max_depth': [None]}
grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=3, scoring='f1', n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
print("Best Random Forest Model:", best_rf)

# Model 3: XGBoost
param_grid_xgb = {'learning_rate': [0.1], 'max_depth': [3], 'n_estimators': [100]}
grid_xgb = GridSearchCV(XGBClassifier(eval_metric='logloss'), param_grid_xgb, cv=3, scoring='f1', n_jobs=-1)
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_
print("Best XGBoost Model:", best_xgb)

# Model Comparison (Cross-Validation)
models = {
    'Logistic Regression': best_lr,
    'Random Forest': best_rf,
    'XGBoost': best_xgb
}

scores = [cross_val_score(model, X_train, y_train, cv=3, scoring='f1', n_jobs=-1).mean() for model in models.values()]

# Plotting the F1 scores of each model
plt.bar(models.keys(), scores)
plt.title('Model F1 Score Comparison (Cross-Validated)')
plt.ylabel('F1 Score (CV=3)')
plt.ylim(0.9, 1.01)
plt.show()

# Error Analysis
# Generating predictions using the best model (Random Forest)
y_pred = best_rf.predict(X_test)

# Identifying misclassified samples
misclassified_indices = np.where(y_test.values != y_pred)[0]
X_test_df = pd.DataFrame(X_test.toarray())  # Ensure this matches how you transform features
wrong_samples = X_test_df.iloc[misclassified_indices]

# Extracting misclassified original texts
original_texts = df_combined.iloc[y_test.index[misclassified_indices]]
misclassified_samples = original_texts[['text', 'label']]

# Print misclassified samples
print("Misclassified samples:\n", misclassified_samples.head(10))

# Creating a word cloud for the misclassified samples
text = ' '.join(misclassified_samples['text'].values) 
wordcloud = WordCloud(max_words=100, background_color='white').generate(text)

# Displaying the word cloud
plt.figure(figsize=(8, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
