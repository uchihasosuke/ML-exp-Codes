# Importing necessary libraries
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Creating a simple dataset
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, 
                            n_informative=2, n_redundant=0, random_state=42)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating and training the Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Making predictions on the test data
y_pred = model.predict(X_test)

# Output: Accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
