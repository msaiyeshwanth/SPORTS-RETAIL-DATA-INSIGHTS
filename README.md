# SPORTS-RETAIL-DATA-INSIGHTS

import plotly.graph_objects as go
import numpy as np

# Example feature importance values obtained from the previous code
feature_importance = np.array([0.15, 0.30, 0.10, 0.25, 0.20])

# Corresponding feature names
feature_names = ['temperature', 'humidity', 'pressure', 'wind_speed', 'visibility']

# Mapping feature importance to feature names
feature_importance_dict = dict(zip(feature_names, feature_importance))

# Sorting features by importance
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
sorted_feature_names = [x[0] for x in sorted_features]
sorted_importance_values = [x[1] for x in sorted_features]

# Create a horizontal bar plot
fig = go.Figure(go.Bar(
    x=sorted_importance_values,
    y=sorted_feature_names,
    orientation='h'
))

# Update layout for better readability
fig.update_layout(
    title="Feature Importance",
    xaxis_title="Importance",
    yaxis_title="Features",
    yaxis=dict(tickmode='linear'),
    template='plotly_white'
)

# Show plot
fig.show()
