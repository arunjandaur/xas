import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import DPGMM
from sklearn.cluster import DBSCAN

if __name__ == "__main_":
    np.set_printoptions(threshold=np.nan)
    X = np.array([np.random.normal(loc=3, scale=.15, size=1000)]).T
    X2 = np.array([np.random.normal(loc=6, scale=.5, size=1000)]).T
    data = np.vstack((np.hstack((X, X2, 2*X)), np.hstack((X, X2, 4*X))))
    num = 300
    db = DBSCAN(eps=.5, min_samples=num)
    db.fit(data)
    plt.plot(data[:, 0], data[:, 1], 'bo')
    plt.show()

    print db.components_
    print db.labels_

if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan)
    X = np.array([np.random.normal(loc=3, scale=.15, size=1000)]).T
    X2 = np.array([np.random.normal(loc=6, scale=.5, size=1000)]).T
    data = np.vstack((np.hstack((X, X2, 2*X)), np.hstack((X, X2, 4*X))))
    num = 300

    for i in range(1, 15):
        db = DBSCAN(eps=.1*i, min_samples=num)
        db.fit(data)
        print db.labels_
        raw_input("Press enter when ready")

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
