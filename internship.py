import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("Advertising.csv")

# Display first rows
print("First 5 Rows:")
print(df.head())

# Remove unnecessary column if present
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# Features and target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMean Absolute Error:", mae)
print("R2 Score:", r2)

# -----------------------------
# Graph 1: Actual vs Predicted
# -----------------------------
plt.figure(figsize=(8,6))

plt.scatter(y_test, predictions)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.show()

# -----------------------------
# Graph 2: Correlation Heatmap
# -----------------------------
plt.figure(figsize=(8,6))

sns.heatmap(df.corr(), annot=True)

plt.title("Correlation Heatmap")

plt.show()

# -----------------------------
# Sample Prediction
# -----------------------------
sample = [[230.1, 37.8, 69.2]]

predicted_sales = model.predict(sample)

print("\nPredicted Sales:", predicted_sales[0])