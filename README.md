# SPORTS-RETAIL-DATA-INSIGHTS

import pandas as pd
import numpy as np

# Example: Creating a DataFrame with hourly data for 5 months and 15 feature columns
date_rng = pd.date_range(start='2024-01-01 00:00', end='2024-06-01 23:00', freq='H')
data = np.random.randn(len(date_rng), 15) * 10 + 100  # Replace with your actual data
columns = [f'feature_{i}' for i in range(1, 16)]
df = pd.DataFrame(data, index=date_rng, columns=columns)

# Step 1: Filter data to include only from 7 AM on January 1st to 7 PM on May 30th
df = df[(df.index >= '2024-01-01 07:00') & (df.index <= '2024-05-30 19:00')]

# Resample to 12-hour intervals, starting from 7 AM
df_resampled = df.resample('12H', origin='start', offset=7, label='right', closed='right').mean()

# Format Timestamps for the new DataFrame
def format_timestamp(index):
    return index.strftime('%Y %m %d %H %M%S')

df_resampled.index = df_resampled.index.map(format_timestamp)

# Reset index to make formatted timestamp a column
df_resampled.reset_index(inplace=True)
df_resampled.rename(columns={'index': 'formatted_timestamp'}, inplace=True)

print(df_resampled.head(10))
