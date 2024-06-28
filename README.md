# SPORTS-RETAIL-DATA-INSIGHTS

ggplot(df, aes(x = indices)) +
  geom_line(aes(y = actual, color = "Actual"), size = 1) +
  geom_line(aes(y = predicted, color = "Predicted"), linetype = "dashed", size = 1) +
  labs(x = "Index", y = "Values", title = "Actual vs Predicted") +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal()

