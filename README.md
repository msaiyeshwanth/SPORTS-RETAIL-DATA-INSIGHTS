# SPORTS-RETAIL-DATA-INSIGHTS

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Step 1: Data Preparation

def prepare_data(df):
    \"\"\"
    Clean and scale the data.
    :param df: DataFrame containing 'pressure', 'oxygen', and 'glass_temperature'.
    :return: Scaled DataFrame.
    \"\"\"
    # Convert timestamp to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Scale features to have mean=0 and variance=1
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['pressure', 'oxygen', 'glass_temperature']])
    
    # Create DataFrame for scaled features
    scaled_df = pd.DataFrame(scaled_features, columns=['pressure', 'oxygen', 'glass_temperature'])
    return scaled_df

def create_sequences(data, target_col='glass_temperature', seq_length=24):
    \"\"\"
    Create input sequences and corresponding targets.
    :param data: Scaled DataFrame.
    :param target_col: Column name for the target variable.
    :param seq_length: Length of each input sequence.
    :return: Input sequences and targets as tensors.
    \"\"\"
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i + seq_length].values  # Get the sequence
        target = data.iloc[i + seq_length][target_col]  # Get the target
        sequences.append(seq)
        targets.append(target)
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

# Load your data
df = pd.read_csv('your_data.csv')  # Replace with your actual file
scaled_df = prepare_data(df)

# Create sequences
X, y = create_sequences(scaled_df)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Model Definition

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GatedResidualNetwork, self).__init__()
        # Two linear layers for the gated residual network
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.glu = nn.GLU()  # Gated Linear Unit for non-linearities
        self.layer_norm = nn.LayerNorm(output_dim)  # Layer normalization

    def forward(self, x):
        hidden = F.elu(self.fc1(x))  # Apply ELU activation
        output = self.fc2(hidden)  # Output from the second layer
        gated_output = self.glu(output)  # Apply GLU
        return self.layer_norm(x + gated_output)  # Add residual connection and normalize

class StaticCovariateEncoder(nn.Module):
    def __init__(self, static_input_dim, hidden_dim):
        super(StaticCovariateEncoder, self).__init__()
        # Encoder for static features
        self.fc = nn.Linear(static_input_dim, hidden_dim)

    def forward(self, static_input):
        return F.elu(self.fc(static_input))  # Apply ELU activation

class TFT(nn.Module):
    def __init__(self, input_size, static_size, lstm_hidden_size, attn_heads, dropout=0.1):
        super(TFT, self).__init__()
        # Static Covariate Encoder
        self.static_encoder = StaticCovariateEncoder(static_size, lstm_hidden_size)

        # Variable Selection Networks
        self.static_var_select = GatedResidualNetwork(input_size, lstm_hidden_size, input_size)
        self.temporal_var_select = GatedResidualNetwork(input_size, lstm_hidden_size, input_size)

        # LSTM layer for sequential processing
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, batch_first=True)

        # Attention layer for long-term dependencies
        self.attn = nn.MultiheadAttention(embed_dim=lstm_hidden_size, num_heads=attn_heads, batch_first=True)

        # Gated and Residual connections
        self.gate = GatedResidualNetwork(lstm_hidden_size, lstm_hidden_size, lstm_hidden_size)

        # Quantile outputs for prediction intervals (10th, 50th, and 90th percentiles)
        self.quantile_outputs = nn.Linear(lstm_hidden_size, 3)
        self.attn_weights = None  # Variable to store attention weights

    def forward(self, static_inputs, temporal_inputs):
        # Encode static inputs
        static_encoded = self.static_encoder(static_inputs)

        # Apply variable selection to static and temporal inputs
        static_selected = self.static_var_select(static_encoded)
        temporal_selected = self.temporal_var_select(temporal_inputs)

        # LSTM for temporal processing
        lstm_out, _ = self.lstm(temporal_selected)

        # Apply multi-head attention
        attn_out, self.attn_weights = self.attn(lstm_out, lstm_out, lstm_out)  # Store attention weights

        # Gated residual network
        gated_out = self.gate(attn_out)

        # Quantile outputs for prediction intervals
        quantile_preds = self.quantile_outputs(gated_out)
        return quantile_preds

# Step 3: Loss Function for Quantile Regression

def quantile_loss(y_true, y_pred, quantiles):
    \"\"\"
    Calculate quantile loss for multiple quantiles.
    :param y_true: True values (targets).
    :param y_pred: Predicted values from the model.
    :param quantiles: List of quantiles to calculate the loss for.
    :return: Mean quantile loss across specified quantiles.
    \"\"\"
    losses = []
    for q in quantiles:
        # Compute loss based on quantile
        loss = torch.where(y_true < y_pred[:, 0], (y_pred[:, 0] - y_true) * q, (y_true - y_pred[:, 0]) * (1 - q))
        losses.append(loss)
    return torch.mean(torch.stack(losses))

# Step 4: Model Initialization and Training with Early Stopping

input_size = 3  # Number of input features
static_size = 0  # Number of static features
lstm_hidden_size = 64
attn_heads = 8
model = TFT(input_size=input_size, static_size=static_size, lstm_hidden_size=lstm_hidden_size, attn_heads=attn_heads)

# Loss function and optimizer
quantiles = [0.1, 0.5, 0.9]  # 10th, 50th, 90th percentiles for quantile regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Early stopping parameters
patience = 10
best_loss = float('inf')
patience_counter = 0

# Create a directory to save the model
model_dir = 'tft_model'
os.makedirs(model_dir, exist_ok=True)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train, X_train)  # Using X_train for both static and temporal inputs
    
    # Calculate quantile loss
    loss = quantile_loss(y_train.unsqueeze(1), y_pred, quantiles)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Early stopping
    if loss < best_loss:
        best_loss = loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(model_dir, 'tft_best_model.pth'))  # Save model
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

# Step 5: Validation and Variable Importance

model.eval()
with torch.no_grad():
    y_val_pred = model(X_val, X_val)

# Accessing variable importance through attention weights
# The attention weights can help understand the importance of different input features

# Convert attention weights to numpy for easier handling
attention_weights_np = model.attn_weights.cpu().numpy()

# Example of averaging attention weights across all heads and time steps for variable importance
mean_attention_weights = np.mean(attention_weights_np, axis=1)  # Average over heads
variable_importance = np.mean(mean_attention_weights, axis=0)  # Average over time steps

# Print variable importance
print("Variable Importance (higher is more important):")
print(variable_importance)

# Step 6: Loading the Model and Making Predictions

def load_model(model_path):
    \"\"\"
    Load the trained model from the specified path.
    :param model_path: Path to the saved model.
    :return: Loaded model.
    \"\"\"
    model = TFT(input_size=input_size, static_size=static_size, lstm_hidden_size=lstm_hidden_size, attn_heads=attn_heads)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
# Example of loading the model and making predictions
loaded_model = load_model(os.path.join(model_dir, 'tft_best_model.pth'))

# Prepare new input data for prediction (should be in the same format as the training data)
new_data = torch.tensor([[...]])  # Replace with actual new data tensor, shape should be (1, seq_length, num_features)








feature_names = ['Pressure', 'Oxygen', 'Glass Temperature']  # Adjust as necessary

# Create a DataFrame for plotting
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': variable_importance
})

# Sort the DataFrame by Importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Create the horizontal bar plot
fig = px.bar(importance_df, 
             x='Importance', 
             y='Feature', 
             orientation='h', 
             title='Variable Importance',
             labels={'Importance': 'Importance Score', 'Feature': 'Features'},
             text='Importance')

# Show the plot
fig.show()

# Make predictions
with torch.no_grad():
    predictions = loaded_model(new_data, new_data)  # Use new_data for both static and temporal inputs

# Print predictions
print("Predictions (quantiles):")
print(predictions)















import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Step 1: Data Preparation

def prepare_data(df):
    """
    Clean and scale the data.
    :param df: DataFrame containing 'pressure', 'oxygen', and 'glass_temperature'.
    :return: Scaled DataFrame.
    """
    # Convert timestamp to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['pressure', 'oxygen', 'glass_temperature']])
    
    # Create DataFrame for scaled features
    scaled_df = pd.DataFrame(scaled_features, columns=['pressure', 'oxygen', 'glass_temperature'])
    return scaled_df

def create_sequences(data, target_col='glass_temperature', seq_length=24):
    """
    Create input sequences and corresponding targets.
    :param data: Scaled DataFrame.
    :param target_col: Column name for the target variable.
    :param seq_length: Length of each input sequence.
    :return: Input sequences and targets as tensors.
    """
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i + seq_length].values  # Get the sequence
        target = data.iloc[i + seq_length][target_col]  # Get the target
        sequences.append(seq)
        targets.append(target)
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

# Step 2: Model Definition

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GatedResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.glu = nn.GLU()  # Gated Linear Unit
        self.layer_norm = nn.LayerNorm(output_dim)  # Layer normalization

    def forward(self, x):
        hidden = F.elu(self.fc1(x))  # Apply ELU activation
        output = self.fc2(hidden)  # Output from second layer
        gated_output = self.glu(output)  # Apply GLU
        return self.layer_norm(x + gated_output)  # Add residual connection and normalize

class StaticCovariateEncoder(nn.Module):
    def __init__(self, static_input_dim, hidden_dim):
        super(StaticCovariateEncoder, self).__init__()
        if static_input_dim > 0:
            self.fc = nn.Linear(static_input_dim, hidden_dim)

    def forward(self, static_input):
        if static_input.size(-1) > 0:
            return F.elu(self.fc(static_input))  # Apply ELU activation only if static input size > 0
        return None

class TFT(nn.Module):
    def __init__(self, input_size, static_size, lstm_hidden_size, attn_heads, dropout=0.1):
        super(TFT, self).__init__()
        
        # Static Covariate Encoder (only if static size > 0)
        self.static_size = static_size
        if static_size > 0:
            self.static_encoder = StaticCovariateEncoder(static_size, lstm_hidden_size)

        # Variable Selection Networks
        self.static_var_select = GatedResidualNetwork(input_size, lstm_hidden_size, input_size)
        self.temporal_var_select = GatedResidualNetwork(input_size, lstm_hidden_size, input_size)

        # LSTM layer
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, batch_first=True)

        # Attention layer
        self.attn = nn.MultiheadAttention(embed_dim=lstm_hidden_size, num_heads=attn_heads, batch_first=True)

        # Gating and Residual connections
        self.gate = GatedResidualNetwork(lstm_hidden_size, lstm_hidden_size, lstm_hidden_size)

        # Final output layer
        self.output_layer = nn.Linear(lstm_hidden_size, 1)

    def forward(self, static_inputs, temporal_inputs):
        if self.static_size > 0 and static_inputs.size(-1) > 0:
            # Encode static inputs if provided
            static_encoded = self.static_encoder(static_inputs)
            static_selected = self.static_var_select(static_encoded)
        else:
            static_selected = None

        # Apply variable selection to temporal inputs
        temporal_selected = self.temporal_var_select(temporal_inputs)

        # LSTM for temporal processing
        lstm_out, _ = self.lstm(temporal_selected)

        # Apply multi-head attention
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)

        # Gated residual network
        gated_out = self.gate(attn_out)

        # Output layer for final forecast
        output = self.output_layer(gated_out)
        return output

# Step 3: Loss Function for Quantile Regression

def quantile_loss(y_true, y_pred, quantiles):
    """
    Calculate quantile loss for multiple quantiles.
    :param y_true: True values (targets).
    :param y_pred: Predicted values from the model.
    :param quantiles: List of quantiles to calculate the loss for.
    :return: Mean quantile loss across specified quantiles.
    """
    losses = []
    for q in quantiles:
        loss = torch.where(y_true < y_pred[:, 0], (y_pred[:, 0] - y_true) * q, (y_true - y_pred[:, 0]) * (1 - q))
        losses.append(loss)
    return torch.mean(torch.stack(losses))

# Model Initialization

input_size = 3  # Number of input features (pressure, oxygen, glass_temperature)
static_size = 0  # Number of static features
lstm_hidden_size = 64
attn_heads = 8
model = TFT(input_size=input_size, static_size=static_size, lstm_hidden_size=lstm_hidden_size, attn_heads=attn_heads)

# Rest of the code for training, evaluation, etc.
