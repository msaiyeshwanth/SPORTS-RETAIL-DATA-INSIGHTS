# SPORTS-RETAIL-DATA-INSIGHTS

import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Load and Prepare Data
df = pd.read_csv("your_data.csv")  # Load your dataset
df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert to datetime

# Create an hourly time index (integer-based)
df['time_idx'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() // 3600
df['time_idx'] = df['time_idx'].astype(int)

# Check for missing values and fill them if necessary
df.fillna(method='ffill', inplace=True)

# Scale the features
scaler = StandardScaler()
df[['pressure', 'oxygen', 'humidity', 'wind_speed']] = scaler.fit_transform(df[['pressure', 'oxygen', 'humidity', 'wind_speed']])

# Scale the target variable
df['target'] = scaler.fit_transform(df[['glass_temperature']])

# Step 2: Define TimeSeriesDataSet (no splitting, using the entire dataset)
data = TimeSeriesDataSet(
    df,
    time_idx='time_idx',  # Time index
    target='target',  # Glass temperature (target)
    group_ids=['glass_id'],  # Single glass sample (group ID)
    max_encoder_length=24,  # Input sequence length (look back 24 time steps)
    max_prediction_length=1,  # Prediction length (predict 1 step ahead)
    static_categoricals=[],  # No static categorical features
    static_reals=[],  # No static real features
    time_varying_known_reals=['pressure', 'oxygen', 'humidity', 'wind_speed'],  # Known features
    time_varying_unknown_reals=['glass_temperature'],  # The target
)

# Step 3: Convert the entire dataset to a DataLoader
dataloader = data.to_dataloader(batch_size=16, shuffle=False, num_workers=4)

# Step 4: Define TFT Model
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

# Step 5: Define Callbacks for Model Checkpointing and Early Stopping
checkpoint_callback = ModelCheckpoint(
    monitor="train_loss",  # Monitoring the training loss
    dirpath="checkpoints",
    filename="tft-{epoch:02d}-{train_loss:.2f}",
    save_top_k=3,
    mode="min",
)

early_stopping_callback = EarlyStopping(
    monitor="train_loss",
    patience=5,
    verbose=True,
    mode="min",
)

# Step 6: Initialize Trainer
trainer = Trainer(
    max_epochs=30,  # Train for 30 epochs
    gpus=1 if torch.cuda.is_available() else 0,  # Use GPU if available
    callbacks=[checkpoint_callback, early_stopping_callback],  # Add callbacks
)

# Step 7: Train the Model on the Entire Dataset
trainer.fit(tft, train_dataloaders=dataloader)

# Step 8: Predictions and Global Variable Importance
# Use the same dataloader to get predictions
predictions, x = tft.predict(dataloader, return_x=True)

# Interpret the global variable importance (over the entire dataset)
interpretation = tft.interpret_output(predictions, reduction="sum")  # Summarize over the entire dataset
tft.plot_variable_importance(interpretation)  # Plot variable importance for the entire dataset

# Step 9: Extract Variable Importance for Specific Data Points

# Example: Get variable importance for the first 5 rows of the dataloader
for i, batch in enumerate(dataloader):
    if i == 5:  # Stop after inspecting 5 batches
        break
    predictions_for_batch, x_for_batch = tft.predict(batch, return_x=True)
    
    # Interpret variable importance for this specific batch
    interpretation_for_batch = tft.interpret_output(predictions_for_batch, reduction="sum")
    
    # Print or plot variable importance for this batch (specific rows)
    print(f"Variable importance for batch {i+1}:")
    print(interpretation_for_batch)
    # Optionally, you can also plot the importance for the batch
    tft.plot_variable_importance(interpretation_for_batch)

# Step 10: Additional Evaluation and Plotting (Optional)
# Convert predictions and actuals back to unscaled values (optional)
# test_predictions = scaler.inverse_transform(predictions)
# actuals = scaler.inverse_transform(x['decoder_target'].numpy())
