# SPORTS-RETAIL-DATA-INSIGHTS

import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load your data
data = pd.read_csv('your_data.csv')

# Assuming 'timestamp' column exists and 'feature1' is the feature of interest
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

# Select the feature of interest
feature_data = data['feature1']

# Define the window size
window_size = 60  # 60-minute moving window

# Calculate moving averages and moving standard deviations
moving_average = feature_data.rolling(window=window_size).mean()
moving_std_dev = feature_data.rolling(window=window_size).std()

# Calculate upper and lower control limits
upper_control_limit = moving_average + 3 * moving_std_dev
lower_control_limit = moving_average - 3 * moving_std_dev

# Create the control chart using Plotly
fig = go.Figure()

# Add the actual data
fig.add_trace(go.Scatter(
    x=feature_data.index,
    y=feature_data,
    mode='lines',
    name='Feature 1',
    line=dict(color='blue')
))

# Add the moving average
fig.add_trace(go.Scatter(
    x=moving_average.index,
    y=moving_average,
    mode='lines',
    name='Moving Average',
    line=dict(color='green')
))

# Add the upper control limit
fig.add_trace(go.Scatter(
    x=upper_control_limit.index,
    y=upper_control_limit,
    mode='lines',
    name='Upper Control Limit',
    line=dict(color='red', dash='dash')
))

# Add the lower control limit
fig.add_trace(go.Scatter(
    x=lower_control_limit.index,
    y=lower_control_limit,
    mode='lines',
    name='Lower Control Limit',
    line=dict(color='red', dash='dash')
))

# Add titles and labels
fig.update_layout(
    title='Control Chart for Feature 1',
    xaxis_title='Time',
    yaxis_title='Feature 1 Value',
    legend_title='Legend',
    hovermode='x unified'
)

# Show the plot
fig.show()



