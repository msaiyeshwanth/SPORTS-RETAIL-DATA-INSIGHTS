# SPORTS-RETAIL-DATA-INSIGHTS

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import shap
import joblib
import numpy as np

# Sample data loading (replace with your own data)
# X = ...  # Features
# y = ...  # Target

# Perform a 70-15-15 split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 0.5 * 0.3 = 0.15

# Define parameter grid for XGBoost
param_grid_xgb = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'alpha': [0, 0.1, 1],
    'lambda': [0, 0.1, 1]
}

# Grid Search for XGBoost
xgb_model = xgb.XGBRegressor(random_state=42)
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=5, n_jobs=-1, scoring='r2')
grid_search_xgb.fit(X_train, y_train)

print(f"Best parameters for XGBoost: {grid_search_xgb.best_params_}")

# Train XGBoost with best parameters
best_xgb_model = grid_search_xgb.best_estimator_
best_xgb_model.fit(X_train, y_train, 
                   eval_set=[(X_val, y_val)], 
                   eval_metric='rmse',
                   verbose=True)

# Predictions and evaluation for XGBoost
y_pred_xgb = best_xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost MSE: {mse_xgb}")
print(f"XGBoost MAE: {mae_xgb}")
print(f"XGBoost R-squared: {r2_xgb}")

# Feature importance for XGBoost
xgb.plot_importance(best_xgb_model, importance_type='weight')
plt.title('XGBoost Feature Importance')
plt.show()

# SHAP values for XGBoost
explainer_xgb = shap.Explainer(best_xgb_model)
shap_values_xgb = explainer_xgb(X_test)
shap.summary_plot(shap_values_xgb, X_test, feature_names=X.columns)

# Save the XGBoost model and SHAP explainer
joblib.dump(best_xgb_model, 'best_xgb_model.pkl')
joblib.dump(explainer_xgb, 'xgb_shap_explainer.pkl')

# LightGBM Model
param_grid_lgb = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.01, 0.1, 0.2],
    'num_leaves': [31, 63, 127],
    'max_depth': [-1, 5, 10],
    'lambda_l1': [0, 0.1, 1],
    'lambda_l2': [0, 0.1, 1]
}

# Grid Search for LightGBM
lgb_model = lgb.LGBMRegressor(random_state=42)
grid_search_lgb = GridSearchCV(estimator=lgb_model, param_grid=param_grid_lgb, cv=5, n_jobs=-1, scoring='r2')
grid_search_lgb.fit(X_train, y_train)

print(f"Best parameters for LightGBM: {grid_search_lgb.best_params_}")

# Train LightGBM with best parameters
best_lgb_model = grid_search_lgb.best_estimator_
best_lgb_model.fit(X_train, y_train, 
                   eval_set=[(X_val, y_val)], 
                   eval_metric='rmse',
                   early_stopping_rounds=50, 
                   verbose=True)

# Predictions and evaluation for LightGBM
y_pred_lgb = best_lgb_model.predict(X_test)
mse_lgb = mean_squared_error(y_test, y_pred_lgb)
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
r2_lgb = r2_score(y_test, y_pred_lgb)

print(f"LightGBM MSE: {mse_lgb}")
print(f"LightGBM MAE: {mae_lgb}")
print(f"LightGBM R-squared: {r2_lgb}")

# Feature importance for LightGBM
lgb.plot_importance(best_lgb_model, importance_type='split')
plt.title('LightGBM Feature Importance')
plt.show()

# SHAP values for LightGBM
explainer_lgb = shap.Explainer(best_lgb_model)
shap_values_lgb = explainer_lgb(X_test)
shap.summary_plot(shap_values_lgb, X_test, feature_names=X.columns)

# Save the LightGBM model and SHAP explainer
joblib.dump(best_lgb_model, 'best_lgb_model.pkl')
joblib.dump(explainer_lgb, 'lgb_shap_explainer.pkl')
