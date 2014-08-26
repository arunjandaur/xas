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
