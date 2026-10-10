
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load Dataset
dataset = pd.read_csv(
    "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
)

dataset.info()

X = dataset.drop('medv', axis=1)
y = dataset['medv']

# SINGLE LINEAR REGRESSION

x = X[['age']]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=10
)

lr = LinearRegression()
lr.fit(x_train, y_train)

result = lr.predict(x_test)

mse = mean_squared_error(y_test, result)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, result)

print("\nSingle Linear Regression:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-Squared: {r2:.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(y_test, result)
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    linestyle='--',
    color='red'
)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Single Linear Regression')
plt.show()


# MULTIPLE LINEAR REGRESSION

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

lr = LinearRegression()
lr.fit(x_train, y_train)

result1 = lr.predict(x_test)

mse = mean_squared_error(y_test, result1)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, result1)

print("\nMultiple Linear Regression:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-Squared: {r2:.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(y_test, result1)
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    linestyle='-',
    color='red'
)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Multiple Linear Regression')
plt.show()
