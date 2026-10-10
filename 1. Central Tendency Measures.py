
import pandas as pd
import numpy as np

data = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
)

mn = np.mean(data["sepal_length"])
md = np.median(data["sepal_length"])
mod = data["sepal_length"].mode()
sd = np.std(data["sepal_length"])
var = np.var(data["sepal_length"])

print("mean:", mn)
print("md:", md)
print("mod:", mod)
print("SD:", sd)
print("Var:", var)
