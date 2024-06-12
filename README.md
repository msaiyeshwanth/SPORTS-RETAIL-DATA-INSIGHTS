# SPORTS-RETAIL-DATA-INSIGHTS
# Plot the results using Plotly
plot <- plot_ly(dt, x = ~x) %>%
  add_markers(y = ~y, name = "Actual", marker = list(color = 'blue')) %>%
  add_lines(y = ~predicted, name = "Predicted", line = list(color = 'red')) %>%
  layout(title = "Decision Tree Regression",
         xaxis = list(title = "x"),
         yaxis = list(title = "y"),
         width = 800,  # Set the width
         height = 600)  # Set the height

# Display the plot
plot
