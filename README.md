# SPORTS-RETAIL-DATA-INSIGHTS


import pickle
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd

# Load the saved XGBoost model
with open('xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the saved SHAP explainer
with open('shap_explainer.pkl', 'rb') as file:
    explainer = pickle.load(file)

# Load your data (example code)
# Assuming X_test and y_test are already prepared
X_test_df = pd.DataFrame(X_test, columns=feature_names)

# Make predictions
y_pred = model.predict(X_test)

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()



# Calculate SHAP values for the test set
shap_values = explainer(X_test_df)

# SHAP summary plot
shap.summary_plot(shap_values, X_test_df)

# SHAP dependence plot for the first feature
shap.dependence_plot(0, shap_values.values, X_test_df)

# Explain a specific prediction (e.g., the first test instance)
instance_index = 0
shap.waterfall_plot(shap_values[instance_index])
