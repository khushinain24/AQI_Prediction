
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import random

# Load Dataset
df = pd.read_csv('air_quality.csv')

# Drop rows with missing values
df.dropna(inplace=True)

# Feature Columns (excluding Date, Time)
features = ['CO', 'PT08S1', 'NMHC', 'C6H6', 'PT08S2', 'NOx',
            'PT08S3', 'NO2', 'PT08S4', 'PT08S5', 'T', 'RH', 'AH']

# Simulated AQI as target (for demonstration; real AQI formula is complex)
df['AQI'] = (df['CO'] + df['NO2'] + df['NOx'] + df['C6H6']) / 4

X = df[features].values
y = df['AQI'].values

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Utility functions for decision trees
def mse(y):
    mean_y = np.mean(y)
    return np.mean((y - mean_y) ** 2)

def split_dataset(X, y, feature, threshold):
    left_idx = X[:, feature] <= threshold
    right_idx = X[:, feature] > threshold
    return X[left_idx], y[left_idx], X[right_idx], y[right_idx]

def best_split(X, y):
    best_feature, best_threshold, best_mse = None, None, float('inf')
    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            _, y_left, _, y_right = split_dataset(X, y, feature, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            mse_left = mse(y_left)
            mse_right = mse(y_right)
            total_mse = (len(y_left) * mse_left + len(y_right) * mse_right) / len(y)
            if total_mse < best_mse:
                best_feature, best_threshold, best_mse = feature, threshold, total_mse
    return best_feature, best_threshold

# Decision Tree Node
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def build_tree(X, y, depth=0, max_depth=10, min_samples=5):
    if len(y) <= min_samples or depth >= max_depth:
        return Node(value=np.mean(y))
    feature, threshold = best_split(X, y)
    if feature is None:
        return Node(value=np.mean(y))
    X_left, y_left, X_right, y_right = split_dataset(X, y, feature, threshold)
    left_node = build_tree(X_left, y_left, depth + 1, max_depth, min_samples)
    right_node = build_tree(X_right, y_right, depth + 1, max_depth, min_samples)
    return Node(feature=feature, threshold=threshold, left=left_node, right=right_node)

def predict_tree(node, x):
    if node.value is not None:
        return node.value
    if x[node.feature] <= node.threshold:
        return predict_tree(node.left, x)
    else:
        return predict_tree(node.right, x)

# Random Forest
class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            idxs = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X[idxs], y[idxs]
            tree = build_tree(X_sample, y_sample, max_depth=self.max_depth, min_samples=self.min_samples)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([[predict_tree(tree, x) for tree in self.trees] for x in X])
        return predictions.mean(axis=1)

# Train model
rf = RandomForest(n_trees=10, max_depth=8)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluation
mse_score = np.mean((y_test - y_pred) ** 2)
rmse_score = np.sqrt(mse_score)
print(f"MSE: {mse_score:.2f}")
print(f"RMSE: {rmse_score:.2f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test[:100], label='Actual AQI', marker='o')
plt.plot(y_pred[:100], label='Predicted AQI', marker='x')
plt.title('Actual vs Predicted AQI (Sample)')
plt.xlabel('Sample Index')
plt.ylabel('AQI')
plt.legend()
plt.tight_layout()
plt.show()
