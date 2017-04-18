import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")


df = pd.read_csv('assets/436-Web-Mobile-Actions.csv')

array = df.as_matrix(columns=df.columns[1:])

# centers = [[1,1],[5,5],[3,10]]
#
# X, _ = make_blobs(n_samples = 500, centers = centers, cluster_std = 1)
#
# plt.scatter(X[:,0],X[:,1])
# plt.show()

print array

# ms = MeanShift()
# # ms.fit(X)
# ms.fit(df[['web_actions','mobile_actions']])
# labels = ms.labels_
# cluster_centers = ms.cluster_centers_
#
# print(cluster_centers)
# n_clusters_ = len(np.unique(labels))
# print("Number of estimated clusters:", n_clusters_)
#
# colors = 10*['r.','g.','b.','c.','k.','y.','m.']