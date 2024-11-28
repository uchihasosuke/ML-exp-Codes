# Importing necessary libraries
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Sample true labels and predicted labels
y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
y_pred = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1]

# 1. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# 2. Accuracy
accuracy = accuracy_score(y_true, y_pred)

# 3. Precision
precision = precision_score(y_true, y_pred)

# 4. Recall
recall = recall_score(y_true, y_pred)

# 5. F1 Score
f1 = f1_score(y_true, y_pred)

# Output
print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
