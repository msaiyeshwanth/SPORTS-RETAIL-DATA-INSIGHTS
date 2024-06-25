# SPORTS-RETAIL-DATA-INSIGHTS

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import plot_model

# Define a simple LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(10, 1), return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))

# Plot the model architecture
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)







import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model

# Create some dummy data
data = np.random.rand(1, 10, 1)

# Get the outputs of the LSTM layers
lstm_layer_1_output = Model(inputs=model.input, outputs=model.layers[0].output)
lstm_layer_2_output = Model(inputs=model.input, outputs=model.layers[1].output)

# Predict to get the activations
layer_1_activations = lstm_layer_1_output.predict(data)
layer_2_activations = lstm_layer_2_output.predict(data)

# Plot the activations
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.title('LSTM Layer 1 Activations')
plt.imshow(layer_1_activations[0], aspect='auto', cmap='viridis')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title('LSTM Layer 2 Activations')
plt.imshow(layer_2_activations[0].reshape(-1, 1), aspect='auto', cmap='viridis')
plt.colorbar()

plt.show()






# Extract weights from the first LSTM layer
lstm_weights = model.layers[0].get_weights()

# Plot the weights
plt.figure(figsize=(15, 5))
for i, weight_matrix in enumerate(lstm_weights):
    plt.subplot(1, len(lstm_weights), i + 1)
    plt.title(f'Weight Matrix {i + 1}')
    plt.imshow(weight_matrix, aspect='auto', cmap='viridis')
    plt.colorbar()

plt.show()
