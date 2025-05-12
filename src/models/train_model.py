from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, f1_score
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Plot the comparison of F1 scores
plt.bar(models.keys(), scores)
plt.title('Model F1 Score Comparison')
plt.ylabel('F1 Score')
plt.ylim(0.9, 1.01)
plt.show()

# Predictions and confusion matrix for each model

# Logistic Regression Confusion Matrix and F1 Score
y_pred_lr = best_lr.predict(X_test)
cm_lr = confusion_matrix(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
print(f"Logistic Regression F1 Score: {f1_lr}")

plt.figure(figsize=(6, 4))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix for Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Random Forest Confusion Matrix and F1 Score
y_pred_rf = best_rf.predict(X_test)
cm_rf = confusion_matrix(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
print(f"Random Forest F1 Score: {f1_rf}")

plt.figure(figsize=(6, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# XGBoost Confusion Matrix and F1 Score
y_pred_xgb = best_xgb.predict(X_test)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
print(f"XGBoost F1 Score: {f1_xgb}")

plt.figure(figsize=(6, 4))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix for XGBoost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Error analysis with Random Forest
misclassified_indices = np.where(y_test != y_pred_rf)[0]
misclassified_samples = df_combined.iloc[misclassified_indices][['text', 'label']]
print("Misclassified samples:\n", misclassified_samples.head())
