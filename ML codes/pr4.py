# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Generating synthetic data for regression
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on test data
y_pred = model.predict(X_test)

# Output: Coefficients and Intercept
print(f"Coefficient: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Plotting the regression line and data points
plt.scatter(X_test, y_test, color='blue', label='True values')  # True values
plt.plot(X_test, y_pred, color='red', label='Regression line')  # Predicted regression line
plt.title('Linear Regression: True vs Predicted')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()
