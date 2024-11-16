import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

# Load Dataset (e.g., MNIST as an example)
data = fetch_openml('mnist_784', version=1)
X = data.data / 255.0  # Normalize pixel values
y = data.target.astype(int)

# Filter for binary classification (e.g., 3 vs. 8)
X = X[(y == 3) | (y == 8)]
y = y[(y == 3) | (y == 8)]

# Dimensionality Reduction
pca = PCA(n_components=50)  # Reduce dimensions to 50
X_pca = pca.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Optimal K
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    accuracies.append(scores.mean())

optimal_k = k_values[np.argmax(accuracies)]
print(f"Optimal K: {optimal_k}")

# Train Final Model
final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
final_knn.fit(X_train, y_train)
y_pred = final_knn.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Detailed Metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
