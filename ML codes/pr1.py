# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# 1. pandas: Creating a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [24, 27, 22, 32],
        'Salary': [50000, 60000, 55000, 70000]}
df = pd.DataFrame(data)

# Displaying the DataFrame
print("Pandas DataFrame:")
print(df)

# 2. numpy: Creating an array and performing operations
arr = np.array([1, 2, 3, 4, 5])
arr_squared = np.square(arr)
print("\nNumPy Array Squared:")
print(arr_squared)

# 3. scikit-learn: Loading a dataset and standardizing it
iris = load_iris()
X = iris.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScikit-learn: Standardized Iris Dataset:")
print(X_scaled[:5])  # Displaying first 5 rows of the standardized data

# 4. matplotlib: Plotting a simple graph
plt.plot(arr, arr_squared)
plt.title('NumPy Array Squared')
plt.xlabel('Original Values')
plt.ylabel('Squared Values')
plt.show()
