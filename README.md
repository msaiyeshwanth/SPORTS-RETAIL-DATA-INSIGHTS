# SPORTS-RETAIL-DATA-INSIGHTS

# Load the best model
model.load_weights(checkpoint_filepath)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# Make predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

# Reshape y_test and y_pred to be 1D arrays for metric calculations
y_test_flat = y_test.flatten()
y_pred_flat = y_pred.flatten()

# Confusion matrix
cm = confusion_matrix(y_test_flat, y_pred_flat)
print("Confusion Matrix:")
print(cm)

# Other metrics
accuracy = accuracy_score(y_test_flat, y_pred_flat)
precision = precision_score(y_test_flat, y_pred_flat)
recall = recall_score(y_test_flat, y_pred_flat)
f1 = f1_score(y_test_flat, y_pred_flat)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
