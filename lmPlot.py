import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('linear_regression_data.csv')

# Renaming columns for clarity
data.columns = ['X', 'Y']

# Calculating means
mean_x = np.mean(data['X'])
mean_y = np.mean(data['Y'])

# Covariance and variance
cov_xy = np.sum((data['X'] - mean_x) * (data['Y'] - mean_y))
var_x = np.sum((data['X'] - mean_x)**2)

# Calculating slope (b1) and intercept (b0)
b1 = cov_xy / var_x
b0 = mean_y - b1 * mean_x

# Predicting values for plotting
predicted_y = b0 + b1 * data['X']

# Plotting the data points and the linear regression line
plt.scatter(data['X'], data['Y'], color='blue', label='Data points')
plt.plot(data['X'], predicted_y, color='red', label='Regression line')

# Adding labels and title
plt.xlabel('Independent Variable (X)')
plt.ylabel('Dependent Variable (Y)')
plt.title('Linear Regression using Covariance Approach')
plt.legend()

# Display the plot
plt.show()

# Displaying the calculated slope and intercept
print(f'Intercept (b0): {b0}')
print(f'Slope (b1): {b1}')
