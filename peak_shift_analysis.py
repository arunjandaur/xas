from __future__ import division
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from sklearn.cluster import *
from sklearn.mixture import GMM
from math import *
from scipy.optimize import curve_fit
from scipy.stats import norm

def peak_crossing():
	x1 = np.transpose(np.array([list(np.arange(1, 101, 1))]))
	#x2 = np.array([list(np.arange(8, 13, 1))])
	y1 = x1
	z1 = np.transpose(np.array([list(random.random(100))])) * 10.0
	y2 = np.transpose(np.array([list(random.random(100))])) * 10.0
	z2 = -1 * x1

	X = np.hstack((np.vstack((x1, x1)), np.vstack((y1, y2)), np.vstack((z1, z2))))

	g = GMM(n_components=2)
	g.fit(X)
	print g.labels_

def peak_split():
	pass

def gauss(E, sigma, a, b):
	x = E[:, 0]
	energy = E[:, 1]
	A = 1 / (sigma * sqrt(2*pi))
	return A * np.exp(-.5 * np.power((energy - (a*x+b)) / sigma, 2))

if __name__ == "__main__":
	X = np.array([1, 2, 2.5, 3, 4])
	sigma = .5
	a = 12
	b = 0
	E = np.array([[], []])

	for x in X:
		mean = a*x
		xdata = np.linspace(mean-2.5*sigma, mean+2.5*sigma, 1000)
		x_s = [x for _ in range(len(xdata))]
		temp = np.vstack((x_s, xdata))
		E = np.hstack((E, temp))

	E = np.transpose(E)
	I = gauss(E, sigma, a, b)
	noise = random.random(len(I)) * max(I)*.05
	noisyI = I + noise
	noise = np.transpose(np.vstack((np.zeros(len(E)), random.random(len(E)) * .05*max(X))))
	noisyE = E + noise

	fitparams, fitcovariance = curve_fit(gauss, noisyE, noisyI, p0 = [2.5, -6, 5], maxfev=4000)
	plt.plot(noisyE[:, 0], noisyI, label = 'original data')
	plt.plot(E[:, 0], gauss(E, *fitparams) ,'bo',label = "fit curve")
	plt.legend()
