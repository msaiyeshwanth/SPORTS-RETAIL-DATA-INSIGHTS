# SPORTS-RETAIL-DATA-INSIGHTS

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
import joblib
import shap  # SHAP for feature importance analysis

# Step 1: Load and Preprocess Data
data = pd.read_csv('glass_manufacturing_data.csv')
features = data[['airflow', 'pressure', 'temperature']]
target = data['glass_temperature'].values  # Replace with your actual target column name

# Standardize features
scaler = StandardScaler()
normed_features = scaler.fit_transform(features)

# Convert to PyTorch tensors
normed_seqs = torch.tensor(normed_features, dtype=torch.float32)
target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(1)  # Make target a column vector

# Step 2: Define Model Parameters
context_length = normed_features.shape[1]  # Number of features
batch_size = 16  # Adjust based on your data size

# Step 3: Load Pre-trained Time-MoE Model
model = AutoModelForCausalLM.from_pretrained(
    'Maple728/TimeMoE-50M',
    device_map="cpu",  # Use "cuda" for GPU
    trust_remote_code=True,
)

# Step 4: Prepare Dataset
class GlassDataset(torch.utils.data.Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]

# Create DataLoader
dataset = GlassDataset(normed_seqs, target_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Step 5: Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Step 6: Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Step 7: Train the Model
trainer.train()

# Step 8: Evaluate Model
model.eval()  # Switch to evaluation mode
predictions = []
with torch.no_grad():
    for batch in dataloader:
        inputs, _ = batch
        output = model.generate(inputs)
        preds = output[:, -1]  # Get last token predictions
        predictions.append(preds)

predictions = torch.cat(predictions).numpy().flatten()  # Convert to 1D array

# Calculate evaluation metrics
mae = mean_absolute_error(target, predictions)
mape = mean_absolute_percentage_error(target, predictions) * 100
r2 = r2_score(target, predictions)

print(f"MAE: {mae}, MAPE: {mape}%, R²: {r2}")

# Step 9: Plot Actual vs. Predicted
plt.figure(figsize=(10, 5))
plt.plot(target, label='Actual', color='blue')
plt.plot(predictions, label='Predicted', color='orange')
plt.title('Actual vs. Predicted Glass Temperature')
plt.xlabel('Sample Index')
plt.ylabel('Glass Temperature')
plt.legend()
plt.show()

# Step 10: Save the Best Trained Model
model.save_pretrained('best_time_moe_model')
scaler_filename = 'scaler.save'
joblib.dump(scaler, scaler_filename)

# Step 11: Feature Importance Analysis Using SHAP
explainer = shap.Explainer(model, normed_seqs)
shap_values = explainer(normed_seqs)

# Visualize feature importance
shap.summary_plot(shap_values, features)

