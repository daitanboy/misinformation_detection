from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib

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
