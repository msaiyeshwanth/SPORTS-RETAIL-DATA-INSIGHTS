# SPORTS-RETAIL-DATA-INSIGHTS

import plotly.graph_objects as go
import pandas as pd

# R-squared values for each model
r_squared_values = [0.85, 0.78, 0.65, 0.72, 0.90, 0.60]
model_names = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6']

# Create a DataFrame for sorting
df = pd.DataFrame({'Model': model_names, 'R_squared': r_squared_values})
df = df.sort_values(by='R_squared', ascending=False)

# Create the horizontal bar chart
fig = go.Figure(data=[
    go.Bar(
        name='R-squared',
        x=df['R_squared'],
        y=df['Model'],
        orientation='h',
        text=df['R_sqaured'],  # Text to display on the bars
        textposition='auto'  # Position of the text on the bars
    )
])

# Customize the layout
fig.update_layout(
    title='Comparison of R-squared Values Across Models',
    xaxis_title='R-squared',
    yaxis_title='Models',
    xaxis=dict(range=[0, 1]),  # Set x-axis range from 0 to 1
    template='plot​⬤
