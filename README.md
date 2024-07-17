# SPORTS-RETAIL-DATA-INSIGHTS

# Function to assign shift labels based on hour
def assign_shift(hour):
    if 7 <= hour < 19:
        return 'Day Shift'
    else:
        return 'Night Shift'

# Create a shift column
df['shift'] = df.index.hour.map(assign_shift)

# Create a shift_start column to handle shifts that span two days
df['shift_start'] = df.index.to_period('D') + (df['shift'] == 'Night Shift').astype('timedelta64[D]')


# Group by shift_start and shift
shift_grouped = df.groupby(['shift_start', 'shift'])

# Aggregate data by shift (e.g., mean, sum, max, min)
shift_aggregated = shift_grouped['value'].agg(['mean', 'sum', 'max', 'min']).reset_index()





# Function to format timestamps
def format_timestamp(row):
    shift_start = row['shift_start'].strftime('%Y %m %d')
    shift_time = '07 0000' if row['shift'] == 'Day Shift' else '19 0000'
    return f"{shift_start} {shift_time}"

# Apply the function to format timestamps
shift_aggregated['formatted_timestamp'] = shift_aggregated.apply(format_timestamp, axis=1)

# Reorder columns to have the formatted timestamp first
shift_aggregated = shift_aggregated[['formatted_timestamp', 'mean', 'sum', 'max', 'min']]

print(shift_aggregated.head(10))
