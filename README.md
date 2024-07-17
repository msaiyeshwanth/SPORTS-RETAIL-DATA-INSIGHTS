# SPORTS-RETAIL-DATA-INSIGHTS

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetRegressor
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load your data into a DataFrame
data = pd.read_csv('your_shift_data.csv')  # Replace with your actual data file
features = data.drop(columns=['efficiency'])
target = data['efficiency']

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert data to numpy arrays
X_train_scaled = np.array(X_train_scaled)
X_val_scaled = np.array(X_val_scaled)
X_test_scaled = np.array(X_test_scaled)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

# Initialize and train TabNet model
tabnet = TabNetRegressor()
tabnet.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], eval_metric=['rmse'])

# Predict and evaluate the model
y_pred = tabnet.predict(X_test_scaled)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"R-squared: {r2}")

# Get feature importance
feature_importances = tabnet.feature_importances_

# Plot feature importance
plt.barh(features.columns, feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance from TabNet')
plt.show()

# Initialize SHAP explainer
explainer = shap.Explainer(tabnet)

# Calculate SHAP values
shap_values = explainer(X_test_scaled)

# Plot SHAP values
shap.summary_plot(shap_values, X_test_scaled, feature_names=features.columns)

# Save the TabNet model
tabnet.save_model('tabnet_model')

# Save the SHAP explainer
joblib.dump(explainer, 'shap_explainer.pkl')

print("Model and SHAP explainer saved successfully.")
