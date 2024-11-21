from Read_data import GetDataset

dataset = GetDataset().fomated_Data()

#Step 1 get data and formatted
areas = dataset[1]
prices = dataset[0]

def is_numeric(value):
    try:
        float(value)  # Try converting to a float
        return True
    except ValueError:
        return False
    
cleaned_data = [
    (float(area), float(price))
    for area, price in zip(areas, prices)
    if is_numeric(area) and is_numeric(price)
]
cleaned_areas, cleaned_prices = zip(*cleaned_data)


n = len(cleaned_areas)
sum_x = sum(cleaned_areas)
sum_y = sum(cleaned_prices)
sum_xy = sum(x * y for x, y in zip(cleaned_areas, cleaned_prices))
sum_x2 = sum(x ** 2 for x in cleaned_areas)

# Step 2: Calculate slope (m) and intercept (b) using normal equations
m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
b = (sum_y - m * sum_x) / n

# Step 3: Define prediction function
def predict(x):
    return m * x + b

# Step 4: Make predictions
predictions = [predict(x) for x in cleaned_areas]

# Step 5: Print results
print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")
print(f"Predictions: {predictions}")

# Step 6: Optional evaluation (R-squared)
mean_y = sum_y / n
ss_total = sum((yi - mean_y) ** 2 for yi in cleaned_prices)
ss_residual = sum((cleaned_prices[i] - predictions[i]) ** 2 for i in range(n))
r_squared = 1 - (ss_residual / ss_total)
print(f"R-squared: {r_squared}")
