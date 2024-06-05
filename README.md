# SPORTS-RETAIL-DATA-INSIGHTS

def kruskal_wallis_test(data, period='month'):
    data = handle_missing_values(data)
    data = handle_outliers(data)
    
    if period == 'month':
        data['month'] = data.index.month
        groups = [data[column].groupby(data['month']).apply(list) for column in data.columns]
    elif period == 'day_of_week':
        data['day_of_week'] = data.index.dayofweek
        groups = [data[column].groupby(data['day_of_week']).apply(list) for column in data.columns]
    else:
        raise ValueError("Period must be either 'month' or 'day_of_week'")
    
    for i, column in enumerate(data.columns[:-1]):
        kruskal_result = kruskal(*groups[i])
        print(f"Kruskal-Wallis Test for {column} by {period}: H-statistic={kruskal_result.statistic}, p-value={kruskal_result.pvalue}")

# Load your data into a DataFrame `df`
# df = pd.read_csv('your_data.csv', parse_dates=['timestamp_column'], index_col='timestamp_column')

# Example function call
# kruskal_wallis_test(df, period='month')
# kruskal_wallis_test(df, period='day_of_week')
