# SPORTS-RETAIL-DATA-INSIGHTS

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectFromModel
import shap
import joblib
import matplotlib.pyplot as plt

# Example data loading (replace with your actual data)
# X = ...  # Features (DataFrame)
# y = ...  # Target (Series)

# Create example data for demonstration
# Remove this section and use your own data loading method
np.random.seed(42)
X = pd.DataFrame(np.random.rand(1000, 30), columns=[f"feature_{i}" for i in range(30)])
y = pd.Series(np.random.rand(1000))

# Split data into training, validation, and test sets (70-15-15 split)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 0.5 * 0.3 = 0.15

# Define parameter grid for RandomForestRegressor
param_grid_rf = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Grid Search for RandomForestRegressor
rf_model = RandomForestRegressor(random_state=42)
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, scoring='r2')
grid_search_rf.fit(X_train, y_train)

print(f"Best parameters for Random Forest: {grid_search_rf.best_params_}")

# Train RandomForest with best parameters
best_rf_model = grid_search_rf.best_estimator_
best_rf_model.fit(X_train, y_train)

# Feature selection
selector = SelectFromModel(best_rf_model, prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Retrain the model with selected features
best_rf_model.fit(X_train_selected, y_train)

# Predictions and evaluation for RandomForestRegressor
y_pred_rf = best_rf_model.predict(X_test_selected)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"RandomForest MSE: {mse_rf}")
print(f"RandomForest MAE: {mae_rf}")
print(f"RandomForest R-squared: {r2_rf}")

# Feature importance for RandomForest
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X_train_selected.shape[1]):
    print(f"{f + 1}. feature {indices[f]} ({importances[indices[f]]})")

# Plot the feature importances
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train_selected.shape[1]), importances[indices], color="r", align="center")
plt.xticks(range(X_train_selected.shape[1]), indices)
plt.xlim([-1, X_train_selected.shape[1]])
plt.show()

# SHAP values for RandomForest
explainer_rf = shap.Explainer(best_rf_model, X_train_selected)
shap_values_rf = explainer_rf(X_test_selected)

# Summary plot
shap.summary_plot(shap_values_rf, X_test_selected)

# Calculate mean absolute SHAP values for feature importance
shap_values_abs = np.abs(shap_values_rf.values).mean(axis=0)
shap_importance = pd.DataFrame(list(zip(X.columns[selector.get_support()], shap_values_abs)),
                               columns=['Feature', 'SHAP Importance'])

# Sort by SHAP importance
shap_importance = shap_importance.sort_values(by='SHAP Importance', ascending=False)

print(shap_importance)

# Plot SHAP feature importance
plt.figure()
plt.title("SHAP Feature Importance")
plt.barh(shap_importance['Feature'], shap_importance['SHAP Importance'], color='b', align='center')
plt.xlabel("SHAP Importance")
plt.gca().invert_yaxis()
plt.show()

# Save the RandomForest model and SHAP explainer
joblib.dump(best_rf_model, 'best_rf_model.pkl')
joblib.dump(explainer_rf, 'rf_shap_explainer.pkl')
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
