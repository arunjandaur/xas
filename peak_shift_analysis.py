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

def gauss(E, sigma, a):
	x = E[:, 0]
	energy = E[:, 1]
	A = 1 / (sigma * sqrt(2*pi))
	return A * np.exp(-.5 * np.power((energy - a*x) / sigma, 2))

if __name__ == "__main__":
	X = np.array([100, 120, 140, 160, 180, 200, 210, 215, 500])
	sigma = 50
	a = 3
	E = np.array([[], []])

	for x in X:
		mean = a*x
		xdata = np.linspace(mean-200, mean+200, 10000)
		x_s = [x for _ in range(len(xdata))]
		temp = np.vstack((x_s, xdata))
		E = np.hstack((E, temp))

	E = np.transpose(E)
	I = gauss(E, sigma, a)

	fitparams, fitcovariance = curve_fit(gauss, E, I, p0 = [5000, 0])
	plt.plot(E, I, label = 'original data')
	plt.plot(E, gauss(E, *fitparams) ,'bo',label = "fit curve")
	plt.legend()
