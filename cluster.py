import numpy as np
import sklearn as sl
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
import pylab as pl

def plot_clusters(db):
	labels = db.labels_
	core_samples = db.core_sample_indices_
	#Black removed and is used for noise instead.
	unique_labels = set(labels)
	colors = pl.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
	for k, col in zip(unique_labels, colors):
		if k == -1:
		# Black used for noise.
			col = 'k'
			markersize = 6
		class_members = [index[0] for index in np.argwhere(labels == k)]
		cluster_core_samples = [index for index in core_samples if labels[index] == k]
	for index in class_members:
		x = X[index]
		if index in core_samples and k != -1:
			markersize = 14
		else:
			markersize = 6
		pl.plot(x[0], x[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=markersize)

	pl.title('Estimated number of clusters: %d' % n_clusters_)
	pl.show()

def cluster(data):
	data = 1 / data
	coords = []
	for row in data:
		coords.append(tuple(row))
	
	distance_matrix = distance.squareform(distance.pdist(coords))
	dbscan = DBSCAN(metric='precomputed').fit(distance_matrix)
	print dbscan.labels_
	#plot_clusters(dbscan)
