library(forecast)

# Specify the input file name


file_name <- "input.txt"

# Read all input from the file
input_data <- readLines(file_name)

# Parse the first line to get N (the number of rows)
N <- as.integer(input_data[1])

# Initialize vectors to store parsed data
timestamps <- c()
prices <- c()

# Parse the next N rows
for (i in 2:(N + 1)) {
    row <- strsplit(input_data[i], "\t")[[1]]  # Split by tab delimiter
    timestamps <- c(timestamps, row[1])
    prices <- c(prices, row[2])
}

df <- data.frame(Timestamp = timestamps, Price = prices, stringsAsFactors = FALSE)

df$Price <- as.numeric(df$Price)

df$Predicted <- NA






# Load necessary libraries


# Example time series data with missing values
price_data <- df$Price

# Store the indices of missing values
missing_indices <- which(is.na(price_data))

# Fill missing values with the mean for ARIMA model fitting
price_data_filled <- ifelse(is.na(price_data), mean(price_data, na.rm = TRUE), price_data)

# Convert to a time series object
price_ts <- ts(price_data_filled)

# Fit an ARIMA model (order = (1, 1, 1) as in the Python example)
model <- auto.arima(price_ts, d=1, max.p=1, max.q=1)

# Forecast the next values (predict the entire series, including missing values)
predictions <- fitted(model)

# Print only the values that were originally missing
cat("Predicted values for missing data:\n")
cat(predictions[missing_indices], "\n")
