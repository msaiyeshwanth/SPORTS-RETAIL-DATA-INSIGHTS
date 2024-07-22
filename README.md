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

# Separate features and target
features = df.iloc[:, :-1].values
target = df.iloc[:, -1].values

# Normalize features and target
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

features_scaled = feature_scaler.fit_transform(features)
target_scaled = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()

# Create sequences
X, y = [], []
for i in range(len(features_scaled) - seq_length):
    x = features_scaled[i:i + seq_length]
    y.append(target_scaled[i + seq_length])
    X.append(x)
X = np.array(X)
y = np.array(y)

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2 + 0.2, shuffle=False)
val_size_adjusted = 0.2 / (0.2 + 0.2)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, shuffle=False)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoaders
batch_size = 16
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
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

## Function to evaluate the model using DataLoader
def evaluate_model_with_dataloader(model, criterion, data_loader):
    model.eval()
    total_loss = 0
    predictions_list = []
    true_values_list = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            predictions = model(X_batch)
            loss = criterion(predictions[:, -1, :], y_batch)  # Assuming last time step is used
            total_loss += loss.item() * X_batch.size(0)  # Accumulate loss for the batch

            predictions_list.append(predictions[:, -1, :].cpu().numpy())
            true_values_list.append(y_batch.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    predictions_np = np.concatenate(predictions_list, axis=0).flatten()
    y_true_np = np.concatenate(true_values_list, axis=0).flatten()

    return avg_loss, predictions_np, y_true_np

# Get predictions, test loss, and R-squared value
test_loss, predictions_np, y_true_np = evaluate_model_with_dataloader(model, criterion, test_loader)

# Calculate R-squared value
def r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    return 1 - (ss_residual / ss_total)

r2_score = r_squared(y_true_np, predictions_np)

# Print test loss and R-squared value
print(f"Test Loss: {test_loss}")
print(f"R-Squared Value: {r2_score}")

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


import shap

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

# Create a SHAP explainer
def create_shap_explainer(model, data_loader):
    # Get a sample of the data
    sample_data = []
    for x_batch, _ in data_loader:
        sample_data.append(x_batch.numpy())
        if len(sample_data) * batch_size >= 100:  # Adjust sample size as needed
            break
    sample_data = np.concatenate(sample_data, axis=0)
    
    shap_model = SHAPWrapper(model)
    explainer = shap.KernelExplainer(shap_model.predict, sample_data[:100])  # Use a subset for speed
    return explainer

# Create the SHAP explainer
explainer = create_shap_explainer(model, train_loader)

# Select a batch from the test_loader to explain
def get_data_from_loader(data_loader, batch_index=0):
    for i, (x_batch, _) in enumerate(data_loader):
        if i == batch_index:
            return x_batch.numpy()
    return None

# Explain a subset of test data
test_data = get_data_from_loader(test_loader, batch_index=0)
shap_values = explainer.shap_values(test_data[:100])  # Use a subset for speed

# Visualize the SHAP values
shap.summary_plot(shap_values, test_data[:100])




def get_attention_scores(model, data_loader):
    model.eval()
    all_attention_scores = []
    
    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)
            _, attention_scores = model(X_batch)
            attention_scores = attention_scores.cpu().numpy()  # Convert to numpy array
            all_attention_scores.append(attention_scores)
    
    return np.concatenate(all_attention_scores, axis=0)

# Get attention scores for training and test data
train_attention_scores = get_attention_scores(model, train_loader)
test_attention_scores = get_attention_scores(model, test_loader)


def aggregate_attention_scores(attention_scores):
    # Compute mean attention scores for each feature
    avg_attention_scores = np.mean(attention_scores, axis=0)
    return avg_attention_scores

# Aggregate attention scores
train_avg_attention_scores = aggregate_attention_scores(train_attention_scores)
test_avg_attention_scores = aggregate_attention_scores(test_attention_scores)

# Combine training and test scores (optional)
overall_avg_attention_scores = (train_avg_attention_scores + test_avg_attention_scores) / 2



import matplotlib.pyplot as plt

def visualize_attention_scores(attention_scores, feature_names):
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_names)), attention_scores)
    plt.xticks(range(len(feature_names)), feature_names, rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Average Attention Score')
    plt.title('Overall Feature Importance Based on Attention Scores')
    plt.show()



    

# Assuming feature_names is a list of feature names
visualize_attention_scores(overall_avg_attention_scores, feature_names)















# Extract weights from the model
def get_layer_weights(model, layer_name):
    layer = dict(model.named_modules())[layer_name]
    return layer.weight.data.cpu().numpy()

# Compute feature importance from weights
def compute_feature_importance(weights):
    # Compute the importance as the mean absolute value of weights for each feature
    feature_importance = np.mean(np.abs(weights), axis=0)
    return feature_importance

# Visualize feature importance
def visualize_feature_importance(importance_scores, feature_names):
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_names)), importance_scores)
    plt.xticks(range(len(feature_names)), feature_names, rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importance Based on Model Weights')
    plt.show()

# Example feature names
feature_names = [f'Feature {i}' for i in range(30)]  # Adjust based on your actual feature names

# Instantiate the model
input_dim = 30  # Example input dimension
hidden_dim = 64
output_dim = 1
model = TemporalFusionTransformer(input_dim, hidden_dim, output_dim).to('cpu')  # Use 'cuda' if GPU is available

# Load the trained model state (if available)
# model.load_state_dict(torch.load('path_to_trained_model.pth'))

# Extract weights from the VariableSelectionNetwork's linear layer
weights = get_layer_weights(model.variable_selection.fc, 'fc')

# Compute feature importance
feature_importance = compute_feature_importance(weights)

# Visualize feature importance
visualize_feature_importance(feature_importance, feature_names)
