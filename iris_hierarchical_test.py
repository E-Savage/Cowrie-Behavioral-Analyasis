import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.datasets import load_iris


# Load the Iris dataset
iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv').values[:, :-1]

# choose linkage array for clustering performance
linkage_array = linkage(iris, method='average', metric="euclidean")

# Plot the dendrogram
dendrogram(linkage_array)
plt.show()

# extract clusters from linkage array for a certain threshhold and then assign clusters to the data points
cluster_assignments = fcluster(linkage_array, t=1.5, criterion='distance')
print(cluster_assignments)