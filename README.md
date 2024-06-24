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











import lime
import lime.lime_tabular

# Initialize the LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_val.reshape(X_val.shape[0], -1),
    feature_names=[f'feature_{i}' for i in range(X_val_flat.shape[1])],
    class_names=['target'],
    verbose=True,
    mode='regression'
)

# Explain the first instance in the validation set
exp = explainer.explain_instance(X_val_flat[0], model_predict, num_features=10)
exp.show_in_notebook(show_table=True)
