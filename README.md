# SPORTS-RETAIL-DATA-INSIGHTS

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_forecasting.metrics import RMSE, MAE
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Step 1: Data Preparation
# Assuming 'pressure', 'oxygen', 'glass_temperature' are columns in your data

# Example data preparation
df = pd.read_csv("your_data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['time_idx'] = (df['timestamp'] - df['timestamp'].min()).dt.days  # Sequential time index

# Scale the features
scaler = StandardScaler()
df[['pressure', 'oxygen', 'humidity', 'wind_speed']] = scaler.fit_transform(df[['pressure', 'oxygen', 'humidity', 'wind_speed']])

# Set the target variable
df['target'] = df['glass_temperature']

# Step 2: Define TimeSeriesDataSet
max_encoder_length = 1  # Can be set to 6 for another run (1 and 6 sequence lengths)
max_prediction_length = 1  # Since you're analyzing rather than forecasting

# Define the TimeSeriesDataSet
data = TimeSeriesDataSet(
    df,
    time_idx='time_idx',
    target='target',
    group_ids=['glass_id'],  # Just use a dummy group ID for the single glass sample
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],  # No static categorical features
    static_reals=[],  # No static real features
    time_varying_known_reals=['pressure', 'oxygen', 'humidity', 'wind_speed'],  # Features used
    time_varying_unknown_reals=['glass_temperature'],  # Glass temperature as the target variable
)

# Step 3: Create DataLoader
batch_size = 16
train_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

# Step 4: Define TFT Model
tft = TemporalFusionTransformer.from_dataset(
    data,
    learning_rate=0.03,
    hidden_size=16,  # Number of hidden units in the LSTM
    attention_head_size=1,
    dropout=0.2,
    hidden_continuous_size=8,  # Hidden size for continuous variables
    output_size=7,  # Number of quantiles for quantile loss
    loss=QuantileLoss(),  # Quantile loss
    reduce_on_plateau_patience=4,
)

# Step 5: Train the Model
trainer = Trainer(
    max_epochs=30,  # Train for 30 epochs
    gpus=1 if torch.cuda.is_available() else 0,  # Use GPU if available
)

trainer.fit(tft, train_dataloaders=train_dataloader)

# Step 6: Make Predictions
predictions, x = tft.predict(train_dataloader, return_x=True)

# Convert predictions and actuals back to unscaled values
predictions = scaler.inverse_transform(predictions)
actuals = scaler.inverse_transform(x['decoder_target'].numpy())

# Step 7: Evaluation Metrics
r_squared = r2_score(actuals, predictions)
rmse = mean_squared_error(actuals, predictions, squared=False)
mape = mean_absolute_percentage_error(actuals, predictions)

print(f"R-squared: {r_squared:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAPE: {mape:.3f}")

# Step 8: Time Series Plot (Actual vs Predicted)
plt.figure(figsize=(10, 6))
plt.plot(actuals, label='Actual', color='blue')
plt.plot(predictions, label='Predicted', color='red', linestyle='--')
plt.title("Actual vs Predicted Glass Temperature")
plt.xlabel("Time")
plt.ylabel("Glass Temperature")
plt.legend()
plt.show()

# Step 9: Variable Importance
interpretation = tft.interpret_output(predictions, reduction="sum")
tft.plot_variable_importance(interpretation)
