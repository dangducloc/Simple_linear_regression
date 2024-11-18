import numpy as np
from crawling_data import Crawler
from sklearn.linear_model import LinearRegression   
from Read_data import GetDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and format the dataset
obj = GetDataset()
datas = obj.fomated_Data()
areas = np.array(datas[1], dtype=float).reshape(-1, 1)
prices = np.array(datas[0], dtype=float).reshape(-1, 1)

# Split the dataset into training and testing sets (80% train, 20% test)
areas_train, areas_test, prices_train, prices_test = train_test_split(areas, prices, test_size=0.2, random_state=42)

# Create the model
model = LinearRegression()

# Train the model on the training set
model.fit(areas_train, prices_train)

# Get the slope (m) and intercept (b)
slope = model.coef_[0]
intercept = model.intercept_
print(f"Slope (m): {slope}")
print(f"Intercept (b): {intercept}")
print(f"Price = {slope} x Area + {intercept}")

# Make predictions using the trained model
predicted_prices_train = model.predict(areas_train)
predicted_prices_test = model.predict(areas_test)


# Calculate the R-squared score (coefficient of determination) on the test set
test_score = model.score(areas_test, prices_test)
print(f"R-squared score on the test set: {test_score}")

# Calculate and print error metrics for both training and test sets
mae_train = mean_absolute_error(prices_train, predicted_prices_train)
mse_train = mean_squared_error(prices_train, predicted_prices_train)
rmse_train = np.sqrt(mse_train)

mae_test = mean_absolute_error(prices_test, predicted_prices_test)
mse_test = mean_squared_error(prices_test, predicted_prices_test)
rmse_test = np.sqrt(mse_test)

# Display the error metrics for both sets
print("\nTraining Set Error Metrics:")
print(f"MAE: {mae_train:.2f}")
print(f"MSE: {mse_train:.2f}")
print(f"RMSE: {rmse_train:.2f}")

print("\nTest Set Error Metrics:")
print(f"MAE: {mae_test:.2f}")
print(f"MSE: {mse_test:.2f}")
print(f"RMSE: {rmse_test:.2f}")

# Plot the training data, testing data, and regression line
plt.figure(figsize=(10, 6))
plt.grid(visible = True)

# Training data
plt.scatter(areas_train, prices_train, color='blue', label='Training Data')
# Testing data
plt.scatter(areas_test, prices_test, color='green', label='Testing Data')

# Plot the regression line (use both training and test areas for a full line)
plt.plot(areas_train, predicted_prices_train, color='red', label='Regression Line')

# Add labels and title
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Linear Regression: Price vs Area')
plt.legend()

# Show the plot
plt.show()