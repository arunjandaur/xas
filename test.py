import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import DPGMM
from sklearn.cluster import DBSCAN
from main import *

if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan)
    X = np.array([np.random.normal(loc=3, scale=.15, size=300)]).T
    X2 = np.array([np.random.normal(loc=4, scale=.25, size=300)]).T
    data = np.vstack((np.hstack((X, X2, 1.5*X)), np.hstack((X, X2, 2.5*X))))

    inputData, peaks = splitPeakData(data)

    cluster(inputData, peaks)

    #db = DBSCAN(eps=1, min_samples=5)
    #db.fit(data)
    #print db.labels_

    #plt.plot(data[:, 0], data[:, 1], 'bo')
    #plt.show()

if __name__ == "__main_":
    X = np.array([np.random.normal(loc=3, scale=.15, size=1000)]).T
    data = np.vstack((np.hstack((X, 2*X)), np.hstack((X, 4*X))))
    num = 10
    max_iter = 100000
    dpgmm = DPGMM(num, alpha=.1, thresh=.0001, n_iter=max_iter)
    print dpgmm.fit(data)
    plt.plot(data[:, 0], data[:, 1], 'bo')
    plt.show()

    print dpgmm.means_
    print dpgmm.weights_
