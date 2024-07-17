# SPORTS-RETAIL-DATA-INSIGHTS

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import shap
import joblib

# Sample data loading (replace with your own data)
# X = ...  # Features
# y = ...  # Target

# Perform a 70-15-15 split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 0.5 * 0.3 = 0.15

# XGBoost Model
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,  # Number of boosting rounds
    learning_rate=0.01,  # Learning rate
    max_depth=5,        # Maximum depth of trees
    alpha=0.1,         # L1 regularization term
    lambda_=0.1,       # L2 regularization term
    random_state=42    # Seed for reproducibility
)

# Train the XGBoost model
xgb_model.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)],  # Validation set
              eval_metric='rmse',
              early_stopping_rounds=50,  # Stop early if no improvement
              verbose=True)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the XGBoost model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost MSE: {mse_xgb}")
print(f"XGBoost MAE: {mae_xgb}")
print(f"XGBoost R-squared: {r2_xgb}")

# Feature importance
xgb.plot_importance(xgb_model, importance_type='weight')
plt.title('XGBoost Feature Importance')
plt.show()

# SHAP values
explainer_xgb = shap.Explainer(xgb_model)
shap_values_xgb = explainer_xgb(X_test)
shap.summary_plot(shap_values_xgb, X_test, feature_names=X.columns)

# Save the XGBoost model
xgb_model.save_model('xgb_model.json')

# Save SHAP explainer
joblib.dump(explainer_xgb, 'xgb_shap_explainer.pkl')


# LightGBM Model
lgb_model = lgb.LGBMRegressor(
    n_estimators=1000,  # Number of boosting rounds
    learning_rate=0.01,  # Learning rate
    num_leaves=31,      # Number of leaves in full trees
    max_depth=-1,       # Maximum depth of trees
    lambda_l1=0.1,      # L1 regularization term
    lambda_l2=0.1,      # L2 regularization term
    random_state=42    # Seed for reproducibility
)

# Train the LightGBM model
lgb_model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],  # Validation set
              eval_metric='rmse',
              early_stopping_rounds=50,  # Stop early if no improvement
              verbose=True)

# Make predictions
y_pred_lgb = lgb_model.predict(X_test)

# Evaluate the LightGBM model
mse_lgb = mean_squared_error(y_test, y_pred_lgb)
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
r2_lgb = r2_score(y_test, y_pred_lgb)

print(f"LightGBM MSE: {mse_lgb}")
print(f"LightGBM MAE: {mae_lgb}")
print(f"LightGBM R-squared: {r2_lgb}")

# Feature importance
lgb.plot_importance(lgb_model, importance_type='split')
plt.title('LightGBM Feature Importance')
plt.show()

# SHAP values
explainer_lgb = shap.Explainer(lgb_model)
shap_values_lgb = explainer_lgb(X_test)
shap.summary_plot(shap_values_lgb, X_test, feature_names=X.columns)

# Save the LightGBM model
lgb_model.booster_.save_model('lgb_model.txt')

# Save SHAP explainer
joblib.dump(explainer_lgb, 'lgb_shap_explainer.pkl')




from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'alpha': [0, 0.1, 1],
    'lambda': [0, 0.1, 1]
}

grid_search = GridSearchCV(xgb.XGBRegressor(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")




from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.01, 0.1, 0.2],
    'num_leaves': [31, 63, 127],
    'max_depth': [-1, 5, 10],
    'lambda_l1': [0, 0.1, 1],
    'lambda_l2': [0, 0.1, 1]
}

grid_search = GridSearchCV(lgb.LGBMRegressor(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

