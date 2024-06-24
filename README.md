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










# Flatten the validation data for LIME
X_val_flat = X_val.reshape(X_val.shape[0], -1)

# Initialize the LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_val_flat,
    feature_names=[f'feature_{i}' for i in range(X_val_flat.shape[1])],
    class_names=['target'],
    verbose=True,
    mode='regression'
)

# Define a prediction function for LIME
def model_predict(X):
    X_reshaped = X.reshape(X.shape[0], seq_length, X_val.shape[2])
    return model.predict(X_reshaped).flatten()

# Explain the first instance in the validation set
i = 0  # Index of the instance you want to explain
exp = explainer.explain_instance(X_val_flat[i], model_predict, num_features=10)
exp.show_in_notebook(show_table=True)

# To visualize the explanation
exp.as_pyplot_figure()
plt.show()

Explanation of the Code

	1.	Data Preparation:
	•	Load and preprocess the data, including scaling and creating sequences.
	•	Split the data into training and validation sets.
	2.	Model Training:
	•	Define and train an LSTM model using the training data.
	3.	LIME Explanation:
	•	Flatten the validation data to be compatible with LIME.
	•	Initialize a LIME explainer with the flattened validation data.
	•	Define a prediction function that reshapes the input data back to the shape expected by the LSTM model and returns the model’s predictions.
	•	Use LIME to explain a specific instance from the validation set.
	•	Visualize the explanation.

By following these steps, you can use LIME to obtain feature importance for individual predictions made by your LSTM model. This helps in understanding which features contribute most to the model’s predictions.
