# SPORTS-RETAIL-DATA-INSIGHTS
# Calculate R-squared
actual <- dt$y
predicted <- dt$predicted
mean_actual <- mean(actual)
ss_total <- sum((actual - mean_actual)^2)
ss_residual <- sum((actual - predicted)^2)
rsquared <- 1 - (ss_residual / ss_total)

print(paste("R-squared:", round(rsquared, 3)))
