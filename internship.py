import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("Iris.csv")

# Display first rows
print("First 5 Rows:")
print(df.head())

# Remove Id column
df = df.drop("Id", axis=1)

# Features and labels
X = df.drop("Species", axis=1)
y = df["Species"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

# Confusion matrix
cm = confusion_matrix(y_test, predictions)

# -----------------------------
# Graph 1: Confusion Matrix
# -----------------------------
plt.figure(figsize=(6,5))

sns.heatmap(cm, annot=True, fmt='d')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# -----------------------------
# Graph 2: Pairplot
# -----------------------------
sns.pairplot(df, hue="Species")

plt.show()

# -----------------------------
# Sample Prediction
# -----------------------------
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print("\nPredicted Flower Species:", prediction[0])