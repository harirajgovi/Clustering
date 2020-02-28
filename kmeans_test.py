# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 12:30:45 2019

@author: hari4
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv("Mall_Customers.csv")
x_mtx = dataset.iloc[:, -2:].values

#K-Means Clustering
from sklearn.cluster import KMeans

wcss = [KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0).fit(x_mtx).inertia_ for i in range(1, 11)]

#Finding the optimal number of clusters
plt.plot(range(1, 11), wcss, color="green")
plt.title("The Elbow Method (K-cluster identification)")
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.show()

#Predicting the cluster categories by applying fit_predict method to x_mtx
kmeans = KMeans(n_clusters=5, init="k-means++", max_iter=300, n_init=10, random_state=0)
y_cluster = kmeans.fit_predict(x_mtx)

#Visualizing clusters
plt.scatter(x_mtx[y_cluster == 0, 0], x_mtx[y_cluster == 0, 1], s=50, color="blue", label="cautious")
plt.scatter(x_mtx[y_cluster == 1, 0], x_mtx[y_cluster == 1, 1], s=50, color="cyan", label="standard")
plt.scatter(x_mtx[y_cluster == 2, 0], x_mtx[y_cluster == 2, 1], s=50, color="green", label="target")
plt.scatter(x_mtx[y_cluster == 3, 0], x_mtx[y_cluster == 3, 1], s=50, color="red", label="incautious")
plt.scatter(x_mtx[y_cluster == 4, 0], x_mtx[y_cluster == 4, 1], s=50, color="yellow", label="sensible")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, color="brown", label="centroid")
plt.title("Clusters of clients")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()