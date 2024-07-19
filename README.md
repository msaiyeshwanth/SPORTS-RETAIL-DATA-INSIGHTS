# SPORTS-RETAIL-DATA-INSIGHTS


import joblib
import shap
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Load the saved XGBoost model and SHAP explainer
model = joblib.load('xgboost_model.pkl')
explainer = joblib.load('shap_explainer.pkl')

# Load your data (example code)
# Assuming X_test and y_test are already prepared
X_test_df = pd.DataFrame(X_test, columns=feature_names)

# Calculate SHAP values for the test set
shap_values = explainer(X_test_df)

# Select the two instances to compare
index1 = 0  # Replace with your first index
index2 = 1  # Replace with your second index

shap_values_instance1 = shap_values[index1]
shap_values_instance2 = shap_values[index2]

# Add the feature actual values to the DataFrame
comparison_df = pd.DataFrame({
    'feature': feature_names,
    'shap_value_instance1': shap_values_instance1.values,
    'shap_value_instance2': shap_values_instance2.values,
    'value_instance1': X_test_df.iloc[index1].values,
    'value_instance2': X_test_df.iloc[index2].values
})

# Predict the target values for the selected instances
pred_instance1 = model.predict(X_test_df.iloc[[index1]])[0]
pred_instance2 = model.predict(X_test_df.iloc[[index2]])[0]

# Create a bar plot using Plotly
fig = go.Figure()

fig.add_trace(go.Bar(
    x=comparison_df['feature'],
    y=comparison_df['shap_value_instance1'],
    name=f'Instance {index1} (Pred: {pred_instance1:.4f})',
    text=[f'{val:.4f} (val: {comparison_df["value_instance1"][i]:.4f})' for i, val in enumerate(comparison_df['shap_value_instance1'])],
    textposition='auto'
))

fig.add_trace(go.Bar(
    x=comparison_df['feature'],
    y=comparison_df['shap_value_instance2'],
    name=f'Instance {index2} (Pred: {pred_instance2:.4f})',
    text=[f'{val:.4f} (val: {comparison_df["value_instance2"][i]:.4f})' for i, val in enumerate(comparison_df['shap_value_instance2'])],
    textposition='auto'
))

# Update layout for better readability
fig.update_layout(
    title=f'Comparison of SHAP Values for Instances {index1} and {index2}',
    xaxis_title='Feature',
    yaxis_title='SHAP Value',
    template='plotly_white',
    barmode='group'  # Group bars together
)

fig.show()
