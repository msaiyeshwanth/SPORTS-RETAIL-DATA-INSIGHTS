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











# Reshape X_test for SHAP
num_samples = X_val.shape[0]
sequence_length = X_val.shape[1]
num_features = X_val.shape[2]

X_val_reshaped = X_val.reshape(num_samples, sequence_length * num_features)

# Define prediction function for SHAP
def model_predict(data):
    data_reshaped = data.reshape(data.shape[0], sequence_length, num_features)
    return model.predict(data_reshaped)

# Create SHAP explainer
explainer = shap.KernelExplainer(model_predict, X_val_reshaped[:100])

# Compute SHAP values
shap_values = explainer.shap_values(X_val_reshaped[:100], nsamples=100)

# Visualize SHAP values
shap.summary_plot(shap_values, features=X_val_reshaped[:100], feature_names=[f'feature_{i}' for i in range(sequence_length * num_features)])
