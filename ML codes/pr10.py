# Importing necessary libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Creating a simple dataset
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, 
                            n_informative=2, n_redundant=0, random_state=42)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating and training the Support Vector Machine classifier
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# Making predictions on the test data
y_pred = model.predict(X_test)

# Output: Accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
