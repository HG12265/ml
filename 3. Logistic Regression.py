
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load Iris Dataset
df = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
)

# Scatter Plot
plt.figure(figsize=(8, 5))
plt.scatter(
    df.petal_length,
    df.petal_width,
    marker='+',
    color='red'
)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Iris Dataset - Logistic Regression")
plt.show()

# Select Features and Target
x = df[['petal_length', 'petal_width']]
y = df['species']

# Split Dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42
)

# Logistic Regression Model
lr = LogisticRegression(max_iter=200)
lr.fit(x_train, y_train)

# Prediction
result = lr.predict(x_test)

# Prediction Probabilities
probabilities = lr.predict_proba(x_test)

# Model Accuracy
accuracy = lr.score(x_test, y_test)

# Classification Report
classification_rep = classification_report(y_test, result)

print("Logistic Regression Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_rep)
