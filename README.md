# SPORTS-RETAIL-DATA-INSIGHTS

# Create a 4-row subplot figure
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)

# Add line plots for each column
fig.add_trace(go.Scatter(x=df.index, y=df['col1'], mode='lines', name='Column 1'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['col2'], mode='lines', name='Column 2'), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['col3'], mode='lines', name='Column 3'), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['col4'], mode='lines', name='Column 4'), row=4, col=1)

# Update layout
fig.update_layout(height=800, width=800, title_text="Comparison of Four Columns")

# Show the plot
fig.show()
