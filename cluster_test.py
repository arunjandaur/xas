from sklearn.mixture import GMM
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
	points = []
	points.append([1, 1])
	points.append([1, 1.05])
	points.append([1, 1.1])
	points.append([1, 1.15])
        points.append([1, 1.2])
        points.append([1, 1.25])
	points.append([2, 1])
	points.append([2, 1.05])
	points.append([2, 1.1])
	points.append([2, 1.15])
        points.append([2, 1.2])
        points.append([2, 1.25])
	points.append([3, 1.05])
	points.append([3, 1.1])
	points.append([3, 1.15])
	points.append([3, 1.2])
        points.append([3, 1.25])
        points.append([3, 1])
	points = np.array(points)

	g = GMM(3, thresh=.0001, min_covar=.0001, n_iter=2000)
	g.fit(points)
	print g.means_, g.weights_
	plt.plot(points[:, 0], points[:, 1], 'bo')
	plt.show()
