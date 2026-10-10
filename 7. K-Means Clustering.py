
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load Iris Dataset
dataset = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
)

# Select Features
x = dataset.drop('species', axis=1).values

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(x)

# Cluster Labels and Centers
cl = kmeans.labels_
ce = kmeans.cluster_centers_

# Plot Clusters
plt.figure(figsize=(10, 6))

plt.scatter(
    x[:, 0], x[:, 1],
    c=cl, cmap='viridis', s=50
)

# Plot Centroids
plt.scatter(
    ce[:, 0], ce[:, 1],
    c='red', marker='X', s=250
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means Clustering")
plt.show()
