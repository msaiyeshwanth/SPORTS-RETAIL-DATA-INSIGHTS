# SPORTS-RETAIL-DATA-INSIGHTS

# Function to identify outliers using the IQR method
def identify_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (df[column] < lower_bound) | (df[column] > upper_bound)

# Identify outliers in the specified column
outliers = identify_outliers(df, 'column_name')

# Replace outliers with NaN
df.loc[outliers, 'column_name'] = np.nan

# Interpolate NaN values linearly
df['column_name'] = df['column_name'].interpolate(method='linear')

print(df)
