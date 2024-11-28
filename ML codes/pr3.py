# Importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load iris dataset
data = load_iris()
X = data.data
y = data.target

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Output
print("Training Data:")
print(X_train[:5])  # Displaying first 5 rows of training data
print("Training Labels:")
print(y_train[:5])  # Displaying first 5 training labels
print("\nTest Data:")
print(X_test[:5])  # Displaying first 5 rows of test data
print("Test Labels:")
print(y_test[:5])  # Displaying first 5 test labels
