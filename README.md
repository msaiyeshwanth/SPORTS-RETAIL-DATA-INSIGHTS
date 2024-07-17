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

print(df.head())
print(df.tail())

# Step 2: Compute the mean for every 12-hour period
# We will use a rolling window approach for this
window_size = 12  # 12-hour periods
df['mean'] = df.rolling(window=window_size).mean()

# Drop rows with NaN values from the initial period
df = df.dropna()

# We will now aggregate every 12-hour period's means into a new DataFrame
# Extracting the 12-hour period means
df_aggregated = df.groupby(df.index.date).mean()

# Create a new DataFrame with aggregated data
df_final = df_aggregated.copy()

# Format Timestamps for the new DataFrame
def format_timestamp(date):
    return f"{date.strftime('%Y %m %d')} 07 0000"

df_final['formatted_timestamp'] = df_final.index.to_series().apply(format_timestamp)

# Reset index to make formatted timestamp a column
df_final.reset_index(drop=True, inplace=True)

print(df_final.head(10))
