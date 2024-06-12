# SPORTS-RETAIL-DATA-INSIGHTS
# Predicted values
ridge_pred <- predict(ridge_model, x)
lasso_pred <- predict(lasso_model, x)

# Calculate R-squared
ridge_r2 <- 1 - sum((y - ridge_pred)^2) / sum((y - mean(y))^2)
lasso_r2 <- 1 - sum((y - lasso_pred)^2) / sum((y - mean(y))^2)

# Print R-squared values
print(paste("Ridge R-squared:", ridge_r2))
print(paste("Lasso R-squared:", lasso_r2))
