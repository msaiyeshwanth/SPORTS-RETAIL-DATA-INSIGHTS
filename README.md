# SPORTS-RETAIL-DATA-INSIGHTS

import plotly.graph_objects as go

# R-squared values for each model
r_squared_values = [0.85, 0.78, 0.65, 0.72, 0.90, 0.60]
model_names = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6']

# Create the bar chart
fig = go.Figure(data=[
    go.Bar(name='R-squared', x=model_names, y=r_squared_values)
])

# Customize the layout
fig.update_layout(
    title='Comparison of R-squared Values Across Models',
    xaxis_title='Models',
    yaxis_title='R-squared',
    yaxis=dict(range=[0, 1]),  # Set y-axis range from 0 to 1
    template='plotly'
)

# Show the plot
fig.show()
