
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Load Iris Dataset
data = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
)

# Display First 5 Rows
print(data.head())

# Features and Target
x = data.drop('species', axis=1)
y = data['species']

# Split Dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.9, random_state=42
)

# Decision Tree Classification
dt = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3
)

# Train Model
dt.fit(x_train, y_train)

# Predict
r = dt.predict(x_test)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, r))

# Decision Tree Visualization
plt.figure(figsize=(8, 6))

plot_tree(
    dt,
    filled=True,
    feature_names=x.columns,
    class_names=list(dt.classes_),
    rounded=True
)

plt.title("Decision Tree")
plt.show()
