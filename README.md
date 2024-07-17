# SPORTS-RETAIL-DATA-INSIGHTS

import pandas as pd
import numpy as np

# Example: Creating a DataFrame with hourly data for 5 months and 15 feature columns
date_rng = pd.date_range(start='2024-01-01 07:00', end='2024-05-30 19:00', freq='H')
data = np.random.randn(len(date_rng), 15) * 10 + 100  # Replace with your actual data
columns = [f'feature_{i}' for i in range(1, 16)]
df = pd.DataFrame(data, index=date_rng, columns=columns)

# Define shift intervals
def get_shift(timestamp):
    if 7 <= timestamp.hour < 19:
        return 'Day Shift'
    else:
        return 'Night Shift'

# Apply the shift function to the DataFrame
df['shift'] = df.index.to_series().apply(get_shift)

# Group by 'shift' and date, then compute the mean
df_resampled = df.groupby(['shift', df.index.date]).mean()

# Unstack to separate day and night shifts
df_resampled = df_resampled.unstack(level=0)

# Flatten the multi-level column index
df_resampled.columns = [f'{col[0]}_{col[1]}' for col in df_resampled.columns]

# Create formatted timestamps for 7 AM and 7 PM shifts
def format_timestamp(date, shift):
    date_str = date.strftime('%Y-%m-%d')
    if shift == 'Day Shift':
        time_str = '07:00:00'
    else:
        time_str = '19:00:00'
    return f"{date_str} {time_str}"

# Apply the function to generate formatted timestamps
df_resampled['formatted_timestamp'] = df_resampled.index.to_series().apply(lambda x: format_timestamp(x[0], x[1]))

# Reorder columns to have the formatted timestamp first
df_resampled = df_resampled[['formatted_timestamp'] + [col for col in df_resampled.columns if col != 'formatted_timestamp']]

print(df_resampled.head(10))
