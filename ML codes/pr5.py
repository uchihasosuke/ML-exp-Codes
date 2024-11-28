# Importing necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np

# Generating a simple multivariable regression dataset
X, y = make_regression(n_samples=100, n_features=2, noise=10, random_state=42)

# Creating and training the Multivariable Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predicting values
y_pred = model.predict(X)

# Output
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Plotting the results for 2D feature (X1) vs y and (X2) vs y
fig = plt.figure(figsize=(12, 6))


# Plotting predicted vs actual values
ax2 = fig.add_subplot(122)
ax2.scatter(y, y_pred, color='red')
ax2.set_xlabel('Actual y')
ax2.set_ylabel('Predicted y')
ax2.set_title('Actual vs Predicted')

plt.show()
