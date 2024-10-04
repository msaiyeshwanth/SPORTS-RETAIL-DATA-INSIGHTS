# SPORTS-RETAIL-DATA-INSIGHTS

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Step 1: Data Preparation

def prepare_data(df):
    """
    Clean and scale the data.
    :param df: DataFrame containing the input features and target variable.
    :return: Scaled DataFrame.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.drop('target', axis=1))
    
    scaled_df = pd.DataFrame(scaled_features, columns=df.columns[:-1])
    scaled_df['target'] = df['target']
    return scaled_df

def create_sequences(data, target_col='target', seq_length=24):
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
        seq = data.iloc[i:i + seq_length].drop(target_col, axis=1).values  # Get the sequence (features)
        target = data.iloc[i + seq_length][target_col]  # Get the target
        sequences.append(seq)
        targets.append(target)
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

# Step 2: TFT Model Definition

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GatedResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.glu = nn.GLU()
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        hidden = F.elu(self.fc1(x))
        output = self.fc2(hidden)
        gated_output = self.glu(output)
        return self.layer_norm(x + gated_output)

class StaticCovariateEncoder(nn.Module):
    def __init__(self, static_input_dim, hidden_dim):
        super(StaticCovariateEncoder, self).__init__()
        if static_input_dim > 0:
            self.fc = nn.Linear(static_input_dim, hidden_dim)

    def forward(self, static_input):
        if static_input.size(-1) > 0:
            return F.elu(self.fc(static_input))
        return None

class TFT(nn.Module):
    def __init__(self, input_size, static_size, lstm_hidden_size, num_layers, attn_heads, dropout):
        super(TFT, self).__init__()

        # Static Covariate Encoder (only if static size > 0)
        self.static_size = static_size
        if static_size > 0:
            self.static_encoder = StaticCovariateEncoder(static_size, lstm_hidden_size)

        # Variable Selection Networks
        self.temporal_var_select = GatedResidualNetwork(input_size, lstm_hidden_size, input_size)

        # LSTM for temporal input
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)

        # Multihead Attention for long-term dependencies
        self.attn = nn.MultiheadAttention(embed_dim=lstm_hidden_size, num_heads=attn_heads, batch_first=True)

        # Gated and Residual connections
        self.gate = GatedResidualNetwork(lstm_hidden_size, lstm_hidden_size, lstm_hidden_size)

        # Final output layer (Quantile Outputs)
        self.output_layer = nn.Linear(lstm_hidden_size, 3)  # For quantile outputs: 10th, 50th, 90th percentiles

    def forward(self, static_inputs, temporal_inputs):
        if self.static_size > 0 and static_inputs.size(-1) > 0:
            static_encoded = self.static_encoder(static_inputs)
            static_selected = self.temporal_var_select(static_encoded)
        else:
            static_selected = None

        # Temporal input processing
        temporal_selected = self.temporal_var_select(temporal_inputs)

        # LSTM for temporal processing
        lstm_out, _ = self.lstm(temporal_selected)

        # Multi-head attention layer
        attn_out, self.attn_weights = self.attn(lstm_out, lstm_out, lstm_out)

        # Gated residual network
        gated_out = self.gate(attn_out)

        # Output layer for quantile predictions
        output = self.output_layer(gated_out[:, -1, :])  # Only use the last time step for prediction
        return output

# Step 3: Quantile Loss Function

def quantile_loss(y_true, y_pred, quantiles):
    """
    Quantile loss for multiple quantiles.
    :param y_true: True values.
    :param y_pred: Predicted values.
    :param quantiles: List of quantiles for regression.
    :return: Mean quantile loss.
    """
    losses = []
    for q in quantiles:
        loss = torch.where(y_true < y_pred[:, q], (y_pred[:, q] - y_true) * q, (y_true - y_pred[:, q]) * (1 - q))
        losses.append(loss)
    return torch.mean(torch.stack(losses))

# Step 4: Model Initialization

input_size = 5  # Number of input features
static_size = 0  # Assuming no static features
lstm_hidden_size = 64
output_dim = 1
num_layers = 3
attn_heads = 8
dropout = 0.2

model = TFT(input_size=input_size, static_size=static_size, lstm_hidden_size=lstm_hidden_size,
            num_layers=num_layers, attn_heads=attn_heads, dropout=dropout)

# Step 5: Training Loop with Early Stopping and Model Saving

def train_model(model, X_train, y_train, X_val, y_val, quantiles, num_epochs=100, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_loss = float('inf')
    patience_counter = 0
    model_dir = 'tft_model'
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(None, X_train)
        loss = quantile_loss(y_train.unsqueeze(1), y_pred, quantiles)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

        # Early stopping logic
        model.eval()
        with torch.no_grad():
            y_val_pred = model(None, X_val)
            val_loss = quantile_loss(y_val.unsqueeze(1), y_val_pred, quantiles)
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(model_dir, 'tft_best_model.pth'))
            else:
                patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Step 6: Variable Importance Plotting

def plot_variable_importance(model, feature_names):
    """
    Extracts and plots the variable importance using attention weights.
    :param model: Trained TFT model.
    :param feature_names: List of feature names.
    """
    attention_weights = model.attn_weights.detach().cpu().numpy()

    # Average attention weights across heads and time steps
    mean_attention_weights = np.mean(attention_weights, axis=1).mean(axis=0)

    import plotly.express as px
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_attention_weights
    }).sort_values(by='Importance', ascending=False)

    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Variable Importance')
    fig.show()

# Step 7: Example of calling the training function

# Prepare the data and sequences
# Replace 'your_data.csv' with the path to your dataset, adjust column names as needed
# df = pd.read_csv('your_data.csv')
# scaled_df = prepare_data(df)
# X_train, y_train = create_sequences(scaled_df)

quantiles = [0.1, 0.5, 0.9]
# Uncomment after preparing X_train, y_train, X_val, y_val
# train_model(model, X_train, y_train, X_val, y_val, quantiles)

# Step 8: Plot variable importance after training
# Uncomment to plot variable importance after training
# feature_names = ['Pressure', 'Oxygen', 'Glass Temperature', 'Humidity', 'Wind Speed']
# plot_variable_importance(model, feature_names)
