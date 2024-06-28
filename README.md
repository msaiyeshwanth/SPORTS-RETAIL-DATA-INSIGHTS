# SPORTS-RETAIL-DATA-INSIGHTS

plot_ly(data = df, x = ~indices) %>%
  add_lines(y = ~actual, name = 'Actual', line = list(color = 'blue')) %>%
  add_lines(y = ~predicted, name = 'Predicted', line = list(color = 'red', dash = 'dash')) %>%
  layout(title = 'Actual vs Predicted',
         xaxis = list(title = 'Index'),
         yaxis = list(title = 'Values'))
