# Model 1: Logistic Regression
param_grid_lr = {'C': [0.01, 0.1, 1, 10]}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, cv=5, scoring='f1')
grid_lr.fit(X_train, y_train)
best_lr = grid_lr.best_estimator_
best_lr

# Model 2: Random Forest
param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, scoring='f1')
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

# Model 3: XGBoost
param_grid_xgb = {'max_depth': [3, 6, 10], 'learning_rate': [0.01, 0.1, 0.3]}
grid_xgb = GridSearchCV(XGBClassifier(), param_grid_xgb, cv=5, scoring='f1')
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_
