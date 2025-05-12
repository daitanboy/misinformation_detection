from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load train/test data
X_train, X_test, y_train, y_test = joblib.load('train_test_split_data.pkl')
df_combined = pd.read_csv('cleaned_data.csv')

# Logistic Regression
lr_model = GridSearchCV(LogisticRegression(max_iter=1000), {'C': [0.01, 0.1, 1, 10]}, cv=5, scoring='f1')
lr_model.fit(X_train, y_train)
best_lr = lr_model.best_estimator_
print("Best Logistic Regression Model:", best_lr)

# Random Forest
rf_model = GridSearchCV(RandomForestClassifier(), {'n_estimators': [100], 'max_depth': [None]}, cv=3, scoring='f1', n_jobs=-1)
rf_model.fit(X_train, y_train)
best_rf = rf_model.best_estimator_
print("Best Random Forest Model:", best_rf)

# Save the best Random Forest model
joblib.dump(best_rf, 'best_rf_model.pkl')

# XGBoost
xgb_model = GridSearchCV(XGBClassifier(eval_metric='logloss'), {'learning_rate': [0.1], 'max_depth': [3], 'n_estimators': [100]}, cv=3, scoring='f1', n_jobs=-1)
xgb_model.fit(X_train, y_train)
best_xgb = xgb_model.best_estimator_
print("Best XGBoost Model:", best_xgb)

# Cross-validation and F1 score comparison
models = {'Logistic Regression': best_lr, 'Random Forest': best_rf, 'XGBoost': best_xgb}
scores = [cross_val_score(model, X_train, y_train, cv=3, scoring='f1').mean() for model in models.values()]

# Plot the comparison
plt.bar(models.keys(), scores)
plt.title('Model F1 Score Comparison')
plt.ylabel('F1 Score')
plt.ylim(0.9, 1.01)
plt.show()

# Error analysis with Random Forest
y_pred = best_rf.predict(X_test)
misclassified_indices = np.where(y_test != y_pred)[0]
misclassified_samples = df_combined.iloc[misclassified_indices][['text', 'label']]
print("Misclassified samples:\n", misclassified_samples.head())
