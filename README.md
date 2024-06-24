# SPORTS-RETAIL-DATA-INSIGHTS

import shap

# Assuming you have created a KernelExplainer and computed shap_values
explainer = shap.KernelExplainer(model_predict, X_train[:100])
shap_values = explainer.shap_values(X_test[:10])

# Get the mean absolute SHAP values for each feature
mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)

# Print feature importances in descending order
importance_df = pd.DataFrame(list(zip(data.columns, mean_abs_shap_values)), columns=['Feature', 'Importance'])
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(importance_df)
