from __future__ import division
import numpy as np
from sklearn.cluster import *

def peak_crossing():
	x1 = np.array([list(np.arange(1, 100, .5))])
	x2 = np.array([list(np.arange(8, 13, 1))])
	y1 = x1
	y2 = -1 * x1

	X = np.hstack((np.vstack((np.transpose(x1), np.transpose(x1))), np.vstack((np.transpose(y1), np.transpose(y2)))))

	ward = Ward(n_clusters=2)
	ward.fit(X)
	print ward.labels_

if __name__ == "__main__":
	peak_crossing()
