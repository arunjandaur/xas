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

def gauss(E, sigma, a, b, a2):
	x = E[:, 0]
	x2 = E[:, 1]
	energy = E[:, 2]
	A = 1 / (sigma * sqrt(2*pi))
	return A * np.exp(-.5 * np.power((energy - (a*x+b+a2*x2)) / sigma, 2))

if __name__ == "__main__":
	X = np.array([1, 2, 3, 4, 5])
	X2 = np.array([1, 1.4, 2.8, 3.2, 4.5])
	sigma = .5
	a = 3
	b = 3
	a2 = 0
	E = np.array([[], [], []])

	for i in range(len(X)):
		x = X[i]
		x2 = X2[i]
		mean = a*x + b + a2*x2
		xdata = np.linspace(mean-2.5*sigma, mean+2.5*sigma, 1000)
		x_s = [x for _ in range(len(xdata))]
		x2_s = [x2 for _ in range(len(xdata))]
		temp = np.vstack((x_s, x2_s, xdata))
		E = np.hstack((E, temp))

	E = np.transpose(E)
	I = gauss(E, sigma, a, b, a2)
	noise = random.random(len(I)) * max(I)*.05
	noisyI = I + noise
	noise = np.transpose(np.vstack((np.zeros(len(E)), np.zeros(len(E)), random.random(len(E)) * .05*max(X))))
	noisyE = E + noise

	fitparams, fitcovariance = curve_fit(gauss, noisyE, noisyI, p0 = [2.5, 1, 0, 4], maxfev=4000)
	plt.plot(noisyE[:, 0], noisyI, label = 'original data')
	plt.plot(E[:, 0], gauss(E, *fitparams) ,'bo',label = "fit curve")
	plt.legend()
