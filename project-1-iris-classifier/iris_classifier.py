# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()
#print(iris.feature_names)
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target  # 0: setosa, 1: versicolor, 2: virginica

# Basic stats
#print(data.describe())
#print(data['target'].value_counts())

# Visualize: Pairplot for feature relationships
import seaborn as sns  # Bonus: pip install seaborn if you want fancier plots
sns.pairplot(data, hue='target')


# Alternative with Matplotlib: Scatter plot example
plt.figure(figsize=(8,6))
plt.scatter(data['sepal length (cm)'], data['sepal width (cm)'], c=data['target'], cmap='plasma')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Sepal Dimensions by Class')
plt.colorbar(ticks=[0,1,2], format=plt.FuncFormatter(lambda val, loc: iris.target_names[val]))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split data

X = data.drop('target', axis=1)  # Features

y = data['target']  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80/20 split, reproducible

# Scale features (mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Model 1: Logistic Regression
log_reg = LogisticRegression(max_iter=200)  # Increase iter if convergence warning
log_reg.fit(X_train_scaled, y_train)

# Model 2: Decision Tree
tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)  # Limit depth to avoid overfitting
tree_clf.fit(X_train_scaled, y_train)  # Trees don't need scaling, but consistency is good

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

# Predictions
log_pred = log_reg.predict(X_test_scaled)
tree_pred = tree_clf.predict(X_test_scaled)

# Accuracy
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, log_pred):.2f}")
print(f"Decision Tree Accuracy: {accuracy_score(y_test, tree_pred):.2f}")

# Confusion Matrix (example for Logistic)
cm = confusion_matrix(y_test, log_pred)
print("Confusion Matrix:\n", cm)

# Full report
print(classification_report(y_test, log_pred))

# Cross-validation (robust eval)
log_cv = cross_val_score(log_reg, X, y, cv=5)  # 5-fold on full data
print(f"Logistic CV Mean Accuracy: {log_cv.mean():.2f}")

# Bonus: Plot Decision Tree (if you want visual)
from sklearn.tree import plot_tree
plt.figure(figsize=(12,8))
plot_tree(tree_clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()

from sklearn.model_selection import GridSearchCV

# Tune Decision Tree
param_grid = {'max_depth': [2,3,4,5], 'min_samples_split': [2,5,10]}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
print(f"Best Params: {grid_search.best_params_}")
best_tree = grid_search.best_estimator_