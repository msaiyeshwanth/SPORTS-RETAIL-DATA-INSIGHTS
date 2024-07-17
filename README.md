# SPORTS-RETAIL-DATA-INSIGHTS

import xgboost as xgb
from lightgbm import LGBMRegressor

# Using XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5,
    alpha=0.1,
    lambda_=0.1,
    random_state=42
)

xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], eval_metric='rmse', early_stopping_rounds=50)

# Predictions and evaluation
y_pred = xgb_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"R-squared: {r2}")

# Feature importance
import matplotlib.pyplot as plt
xgb.plot_importance(xgb_model, importance_type='weight')
plt.show()

# SHAP values
import shap
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test_scaled)
shap.summary_plot(shap_values, X_test_scaled, feature_names=features.columns)

# Save model
xgb_model.save_model('xgb_model.json')

# Save SHAP explainer
import joblib
joblib.dump(explainer, 'xgb_shap_explainer.pkl')import xgboost as xgb
from lightgbm import LGBMRegressor

# Using XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5,
    alpha=0.1,
    lambda_=0.1,
    random_state=42
)

xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], eval_metric='rmse', early_stopping_rounds=50)

# Predictions and evaluation
y_pred = xgb_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"R-squared: {r2}")

# Feature importance
import matplotlib.pyplot as plt
xgb.plot_importance(xgb_model, importance_type='weight')
plt.show()

# SHAP values
import shap
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test_scaled)
shap.summary_plot(shap_values, X_test_scaled, feature_names=features.columns)

# Save model
xgb_model.save_model('xgb_model.json')

# Save SHAP explainer
import joblib
joblib.dump(explainer, 'xgb_shap_explainer.pkl')
