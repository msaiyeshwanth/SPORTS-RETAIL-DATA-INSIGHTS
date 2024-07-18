# SPORTS-RETAIL-DATA-INSIGHTS


# Choose an index for the specific test vector to explain
instance_index = 0

# Extract the SHAP values and feature values for this instance
instance_shap_values = shap_values_array[instance_index]
instance_features = X_test_df.iloc[instance_index]

# Force plot
shap.force_plot(explainer.expected_value, instance_shap_values, instance_features, feature_names=feature_names, matplotlib=True)
plt.show()

# Waterfall plot (more detailed view)
shap.waterfall_plot(shap.Explanation(values=instance_shap_values, base_values=explainer.expected_value, data=instance_features, feature_names=feature_names))
plt.show()
