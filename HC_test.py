# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 18:35:11 2019

@author: hari4
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv("Mall_Customers.csv")
x_mtx = dataset.iloc[:, -2:].values

#dendogram to identify number of clusters
import scipy.cluster.hierarchy as sch

dendogram = sch.dendrogram(sch.linkage(x_mtx, method='ward'))
plt.title("Dendogram")
plt.xlabel("data points")
plt.ylabel("eucledian distance")
plt.show()

#fitting dataset using aglomerative method
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage='ward')
y_cluster = hc.fit_predict(x_mtx)

#Data Visualization
plt.scatter(x_mtx[y_cluster == 0, 0], x_mtx[y_cluster == 0, 1], s=50, color="blue", label="cautious")
plt.scatter(x_mtx[y_cluster == 1, 0], x_mtx[y_cluster == 1, 1], s=50, color="cyan", label="standard")
plt.scatter(x_mtx[y_cluster == 2, 0], x_mtx[y_cluster == 2, 1], s=50, color="green", label="target")
plt.scatter(x_mtx[y_cluster == 3, 0], x_mtx[y_cluster == 3, 1], s=50, color="red", label="incautious")
plt.scatter(x_mtx[y_cluster == 4, 0], x_mtx[y_cluster == 4, 1], s=50, color="yellow", label="sensible")
plt.title("Hierarchial Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()