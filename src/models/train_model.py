"""
train_model.py
Trains multiple classifiers with GridSearchCV.
"""

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_logistic(X, y):
    param_grid = {'C': [0.01, 0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='f1')
    grid.fit(X, y)
    return grid.best_estimator_

def train_random_forest(X, y):
    param_grid = {'n_estimators': [100], 'max_depth': [10, None]}
    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1')
    grid.fit(X, y)
    return grid.best_estimator_

def train_xgboost(X, y):
    param_grid = {'learning_rate': [0.01, 0.1], 'max_depth': [3, 5], 'n_estimators': [100, 200]}
    grid = GridSearchCV(XGBClassifier(eval_metric='logloss', use_label_encoder=False), param_grid, cv=5, scoring='f1')
    grid.fit(X, y)
    return grid.best_estimator_
