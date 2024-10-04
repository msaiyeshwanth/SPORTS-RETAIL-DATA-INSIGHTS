# SPORTS-RETAIL-DATA-INSIGHTS

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Step 1: Data Preparation for Hourly Data
df = pd.read_csv("your_data.csv")  # Load your dataset
df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert to datetime

# Create an hourly time index (integer-based)
df['time_idx'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() // 3600
df['time_idx'] = df['time_idx'].astype(int)

# Scale the features
scaler = StandardScaler()
df[['pressure', 'oxygen', 'humidity', 'wind_speed']] = scaler.fit_transform(df[['pressure', 'oxygen', 'humidity', 'wind_speed']])

# Set the target variable
df['target'] = df['glass_temperature']

# Step 2: Define TimeSeriesDataSet without group_id
max_encoder_length = 1  # Set to 1 for short sequences; can be set to 6 for testing
max_prediction_length = 1  # We want to predict one step ahead

# Create a dummy group ID column
df['glass_id'] = 0  # Assigning a constant ID for all rows

data = TimeSeriesDataSet(
    df,
    time_idx='time_idx',  # Time index as an integer
    target='target',  # Glass temperature
    group_ids=['glass_id'],  # Dummy group ID for a single glass sample
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],  # No static categorical features
    static_reals=[],  # No static real features
    time_varying_known_reals=['pressure', 'oxygen', 'humidity', 'wind_speed'],  # Known features
    time_varying_unknown_reals=['glass_temperature'],  # Target
)

# Step 3: Sequential Split into Train, Validation, and Test Sets
train_size = int(len(data) * 0.8)  # 80% for training
val_size = int(len(data) * 0.1)  # 10% for validation
test_size = len(data) - train_size - val_size  # Remaining for testing

# Create indices for sequential split
train_indices = list(range(train_size))
val_indices = list(range(train_size, train_size + val_size))
test_indices = list(range(train_size + val_size, len(data)))

# Create subsets
train_data = torch.utils.data.Subset(data, train_indices)
val_data = torch.utils.data.Subset(data, val_indices)
test_data = torch.utils.data.Subset(data, test_indices)

# Step 4: Create DataLoaders without shuffling to preserve time order
batch_size = 16
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=15)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=15)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=15)

# Step 5: Define TFT Model
tft = TemporalFusionTransformer.from_dataset(
    data,
    learning_rate=0.03,
    hidden_size=16,  # Number of hidden units in the LSTM
    attention_head_size=8,
    dropout=0.2,
    hidden_continuous_size=8,
    output_size=3,  # For predicting quantiles (e.g., 10th, 50th, 90th percentiles)
    loss=QuantileLoss(),  # Use Quantile Loss for quantile regression
)

# Step 6: Train the Model
trainer = Trainer(
    max_epochs=30,  # Train for 30 epochs
    gpus=1 if torch.cuda.is_available() else 0,  # Use GPU if available
    log_every_n_steps=5  # Adjust logging interval
)

# Fit the model with the trainer
trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# Step 7: Predictions and Evaluation on the Test Set
test_predictions, x = tft.predict(test_dataloader, return_x=True)

# Convert predictions and actuals back to unscaled values
test_predictions = scaler.inverse_transform(test_predictions)
actuals = scaler.inverse_transform(x['decoder_target'].numpy())

# Step 8: Evaluation Metrics
r_squared = r2_score(actuals, test_predictions)
rmse = mean_squared_error(actuals, test_predictions, squared=False)
mape = mean_absolute_percentage_error(actuals, test_predictions)

print(f"R-squared: {r_squared:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAPE: {mape:.3f}")

# Step 9: Time Series Plot (Actual vs Predicted)
plt.figure(figsize=(10, 6))
plt.plot(actuals, label='Actual', color='blue')
plt.plot(test_predictions, label='Predicted', color='red', linestyle='--')
plt.title("Actual vs Predicted Glass Temperature")
plt.xlabel("Time")
plt.ylabel("Glass Temperature")
plt.legend()
plt.show()

# Step 10: Variable Importance
interpretation = tft.interpret_output(test_predictions, reduction="sum")
tft.plot_variable_importance(interpretation)








import torch
from pytorch_forecasting import TimeSeriesDataSet

# Step 1: Create the TimeSeriesDataSet
data = TimeSeriesDataSet(
    df,
    time_idx='time_idx',  # Time index column (should be integer)
    target='target',  # Glass temperature (your target column)
    group_ids=['glass_id'],  # Group IDs (same for all if a single glass)
    max_encoder_length=24,  # How many time steps to look back
    max_prediction_length=1,  # Predicting one step ahead
    static_categoricals=[],  # Static categorical features
    static_reals=[],  # Static real features
    time_varying_known_reals=['pressure', 'oxygen', 'humidity', 'wind_speed'],  # Known features
    time_varying_unknown_reals=['glass_temperature'],  # The target
)

# Step 2: Sequential Split into Train, Validation, and Test Sets
# Use indices to split the dataset sequentially to prevent data leakage
train_size = int(len(data) * 0.8)  # 80% for training
val_size = int(len(data) * 0.1)  # 10% for validation
test_size = len(data) - train_size - val_size  # Remaining for testing

# Sequential split using the range of indices
train_data = TimeSeriesDataSet.from_dataset(data, stop=train_size)  # Up to train_size
val_data = TimeSeriesDataSet.from_dataset(data, start=train_size, stop=train_size + val_size)  # From train_size to val_size
test_data = TimeSeriesDataSet.from_dataset(data, start=train_size + val_size)  # Remaining data for test set

# Step 3: Convert each subset into a DataLoader using to_dataloader()
train_dataloader = train_data.to_dataloader(batch_size=16, shuffle=False, num_workers=4)
val_dataloader = val_data.to_dataloader(batch_size=16, shuffle=False, num_workers=4)
test_dataloader = test_data.to_dataloader(batch_size=16, shuffle=False, num_workers=4)

# Now you can use train_dataloader, val_dataloader, and test_dataloader in your training loop
