
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Breast Cancer Dataset
data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)

# Encode Diagnosis (Malignant = 1, Benign = 0)
df["diagnosis"] = 1 - data.target

# Features and Target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest Classification
model = RandomForestClassifier(
    n_estimators=100, random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Visualization
plt.figure(figsize=(7, 5))
plt.imshow(cm, cmap="viridis")
plt.title("Random Forest - Breast Cancer")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["Benign (0)", "Malignant (1)"])
plt.yticks([0, 1], ["Benign (0)", "Malignant (1)"])
plt.colorbar()

# Display values inside matrix
for i in range(2):
    for j in range(2):
        plt.text(
            j, i, str(cm[i, j]),
            ha="center", va="center",
            color="white" if cm[i, j] < cm.max()/2 else "black"
        )

plt.tight_layout()
plt.show()
