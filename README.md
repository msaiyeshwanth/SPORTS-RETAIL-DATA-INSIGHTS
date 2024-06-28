# SPORTS-RETAIL-DATA-INSIGHTS

plot(indices, actual, type = 'l', col = 'blue', lty = 1, lwd = 2,
     xlab = "Index", ylab = "Values", main = "Actual vs Predicted")
lines(indices, predicted, col = 'red', lty = 2, lwd = 2)
legend("topright", legend = c("Actual", "Predicted"), col = c("blue", "red"), lty = c(1, 2), lwd = 2)
