# Importing necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Custom dataset: Features (X) and Labels (y)
X = [[2, 3], [1, 4], [4, 2], [3, 3], [5, 1]]
y = [0, 0, 1, 1, 1]

# Creating and training the Random Forest model (with 1 tree for simplicity)
model = RandomForestClassifier(n_estimators=1, random_state=42)
model.fit(X, y)

# Output: Displaying the tree structure from the Random Forest
print("Random Forest Decision Tree Structure:")
plot_tree(model.estimators_[0], filled=True, feature_names=['Feature 1', 'Feature 2'], class_names=['Class 0', 'Class 1'], rounded=True)
plt.show()
