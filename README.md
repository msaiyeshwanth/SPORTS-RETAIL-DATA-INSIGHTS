# SPORTS-RETAIL-DATA-INSIGHTS

import shap

# Reshape X_train and X_test to 2D
X_train_reshaped = X_train.reshape((X_train.shape[0], -1))
X_test_reshaped = X_test.reshape((X_test.shape[0], -1))

# Define a prediction function for the reshaped input
def model_predict(data):
    data_reshaped = data.reshape((data.shape[0], SEQ_LENGTH, X.shape[2]))
    return model.predict(data_reshaped).reshape(-1)

# Calculate SHAP values using KernelExplainer
explainer = shap.KernelExplainer(model_predict, X_train_reshaped[:100])  # Use a subset of training data
shap_values = explainer.shap_values(X_test_reshaped[:10])  # Use a smaller sample of test data
