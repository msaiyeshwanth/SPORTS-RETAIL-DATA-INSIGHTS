# SPORTS-RETAIL-DATA-INSIGHTS


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import shap

# Load data
data = pd.read_csv('path_to_your_data.csv')  # Replace with your data file

# Parameters
sequence_length = 30
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Ensure the ratios sum to 1
assert train_ratio + val_ratio + test_ratio == 1

# Calculate split indices
total_samples = len(data) - sequence_length
train_size = int(total_samples * train_ratio)
val_size = int(total_samples * val_ratio)
test_size = total_samples - train_size - val_size

# Train-validation-test split
train_data = data.iloc[:train_size + sequence_length]
val_data = data.iloc[train_size:train_size + val_size + sequence_length]
test_data = data.iloc[train_size + val_size:train_size + val_size + test_size + sequence_length]

# Normalize data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
val_scaled = scaler.transform(val_data)
test_scaled = scaler.transform(test_data)

# Function to create sequences
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length, :-1]
        y = data[i + seq_length, -1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Create sequences
X_train, y_train = create_sequences(train_scaled, sequence_length)
X_val, y_val = create_sequences(val_scaled, sequence_length)
X_test, y_test = create_sequences(test_scaled, sequence_length)

# Convert to PyTorch tensors and create DataLoaders
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the Gated Linear Units (GLU)
class GLU(nn.Module):
    def __init__(self, input_dim):
        super(GLU, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.fc1(x) * torch.sigmoid(self.fc2(x))

# Define the Variable Selection Network
class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VariableSelectionNetwork, self).__init__()
        self.glu = GLU(input_dim)
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        attention_scores = torch.sigmoid(self.glu(x))
        attended_features = attention_scores * x
        return F.relu(self.fc(attended_features))

# Define the Temporal Fusion Decoder
class TemporalFusionDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout_rate=0.2):
        super(TemporalFusionDecoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=(dropout_rate if num_layers > 1 else 0), bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Hidden dimension is doubled because of bidirection

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        if self.num_layers == 1:
            lstm_out = self.dropout(lstm_out)
        return self.fc(lstm_out)

# Define the Temporal Fusion Transformer model
class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout_rate=0.2):
        super(TemporalFusionTransformer, self).__init__()
        self.variable_selection = VariableSelectionNetwork(input_dim, hidden_dim)
        self.temporal_decoder = TemporalFusionDecoder(hidden_dim, hidden_dim, output_dim, num_layers, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)  # Hidden dimension is doubled because of bidirection

    def forward(self, x):
        selected_features = self.variable_selection(x)
        decoded_features = self.temporal_decoder(selected_features)
        output = self.dropout(decoded_features)
        return self.fc_out(output)
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
input_dim = X_train.shape[2]
hidden_dim = 64
output_dim = 1
dropout_rate = 0.2
num_epochs = 100
patience = 10
best_model_path = 'best_model.pth'

# Initialize model
model = TemporalFusionTransformer(input_dim, hidden_dim, output_dim, dropout_rate).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping and model checkpointing
best_val_loss = float('inf')
epochs_no_improve = 0

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output[:, -1, :], y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}')

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            loss = criterion(output[:, -1, :], y_batch)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}')
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve == patience:
        print('Early stopping!')
        break

# Load the best model
model.load_state_dict(torch.load(best_model_path))

# Evaluate the model
def evaluate_model(model, criterion, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions[:, -1, :], y_test)  # Assuming last time step is used
        return predictions, test_loss

# Example test data (replace with your actual test data)
X_test = torch.randn(9, 20, 10)   # 9 samples, 20 time steps, 10 features
y_test = torch.randn(9, 1)        # 9 samples, 1 target each

# Get predictions and calculate test loss
predictions, test_loss = evaluate_model(model, criterion, X_test, y_test)

# Convert predictions and true values to NumPy arrays
predictions_np = predictions[:, -1, :].numpy().flatten()
y_true_np = y_test.numpy().flatten()

# Print test loss
print(f"Test Loss: {test_loss.item()}")

# Print predictions and actual values
print("Predictions:", predictions_np)
print("Actual Values:", y_true_np)

# Plotting predictions vs. true values
plt.figure(figsize=(10, 6))
plt.scatter(y_true_np, predictions_np, color='blue', label='Predicted vs Actual')
plt.plot([min(y_true_np), max(y_true_np)], [min(y_true_np), max(y_true_np)], color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.grid(True)
plt.show()


# Convert the model to a format SHAP can understand
class SHAPWrapper:
    def __init__(self, model):
        self.model = model
    
    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32).to(device)
            output = self.model(x).cpu().numpy()
        return output[:, -1, 0]  # Adjust based on your output shape

shap_model = SHAPWrapper(model)
explainer = shap.KernelExplainer(shap_model.predict, X_train[:100])  # Use a subset for speed
shap_values = explainer.shap_values(X_test[:100])  # Use a subset for speed

# Visualize the SHAP values
shap.summary_plot(shap_values, X_test[:100])
