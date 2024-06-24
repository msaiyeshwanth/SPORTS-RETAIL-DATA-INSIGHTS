# SPORTS-RETAIL-DATA-INSIGHTS

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# Define the sequence length (e.g., 1440 for one day of data)
sequence_length = 1440

# Create the model
model = Sequential()
model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(sequence_length, 32)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(50, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(50)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)



import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()









# Reshape X_train and X_test to 2D
X_train_reshaped = X_train.reshape((X_train.shape[0], -1))
X_test_reshaped = X_test.reshape((X_test.shape[0], -1))

# Calculate SHAP values using KernelExplainer
explainer = shap.KernelExplainer(model.predict, X_train_reshaped[:100])  # Use a subset of training data
shap_values = explainer.shap_values(X_test_reshaped[:10])  # Use a smaller sample of test data

# Plot the feature importance
shap.summary_plot(shap_values, X_test_reshaped[:10])
