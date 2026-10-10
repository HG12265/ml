
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Load Iris Dataset
data = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
)

# Features and Target
x = data.drop('species', axis=1)
y = data['species']

# Split Dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=32
)

# K-Nearest Neighbors Algorithm
n = KNeighborsClassifier()
n.fit(x_train, y_train)
r1 = n.predict(x_test)

print("Accuracy:", accuracy_score(y_test, r1))

# Gaussian Naive Bayes Algorithm
nb = GaussianNB()
nb.fit(x_train, y_train)
r2 = nb.predict(x_test)

print("Accuracy of Gaussian:", accuracy_score(y_test, r2))

# Encode Class Labels
le = LabelEncoder()
le.fit(y)

# Actual vs Predicted Graph
plt.figure(figsize=(8, 5))
plt.scatter(le.transform(y_test), le.transform(r2))
plt.plot([0, 2], [0, 2], 'r--')

plt.xticks([0, 1, 2], le.classes_)
plt.yticks([0, 1, 2], le.classes_)

plt.title("Gaussian Naive Bayes - Actual VS Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
