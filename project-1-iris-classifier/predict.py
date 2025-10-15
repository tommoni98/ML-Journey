# predict.py - Run with: python predict.py
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Quick train (for demo)
iris = load_iris()
X, y = iris.data, iris.target
scaler = StandardScaler().fit(X)
model = LogisticRegression().fit(scaler.transform(X), y)

# Predict new sample (e.g., [sepal_l, sepal_w, petal_l, petal_w])
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Should be setosa
pred = model.predict(scaler.transform(new_sample))
print(f"Predicted class: {iris.target_names[pred[0]]}")