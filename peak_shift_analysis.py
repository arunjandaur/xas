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

def gauss2(E, sigma1, sigma2, a1, b1, c1, a2, b2, c2):
	return gauss(E, sigma1, a1, b1, c1) + gauss(E, sigma2, a2, b2, c2)

if __name__ == "__main__":
	X = (np.random.normal(loc=1.16, scale=.16, size=10) - 1.16) / .16
	X2 = (np.random.normal(loc=3, scale=.3, size=10) - 3) / .3
	sigma1 = .5
	sigma2 = .55
	a1 = .8
	b1 = 0
	c1 = 1
	a2 = -.8
	b2 = 0
	c2 = 6
	E = np.array([[], [], []])

	for i in range(len(X)):
		x = X[i]
		x2 = X2[i]
		mean = a1*x + b1*x2 + c1
		mean2 = a2*x + b2*x2 + c2
		#interv1 = np.linspace(mean-2.5*sigma1, mean+2.5*sigma1, 10000)
		#interv2 = np.linspace(mean2-2.5*sigma2, mean2+2.5*sigma2, 10000)
		#xdata = np.hstack((interv1, interv2))
		xdata = np.linspace(mean-2.5*sigma1, mean2+2.5*sigma2)
		x_s = [x for _ in range(len(xdata))]
		x2_s = [x2 for _ in range(len(xdata))]
		temp = np.vstack((x_s, x2_s, xdata))
		E = np.hstack((E, temp))

	E = np.transpose(E)
	I = gauss2(E, sigma1, sigma2, a1, b1, c1, a2, b2, c2)
	noise = random.random(len(I)) * np.mean(I)*.025
	noisyI = I + noise
	noise = np.transpose(np.vstack((np.zeros(len(E)), np.zeros(len(E)), random.random(len(E)) * .025*np.mean(E[:,2]))))
	noisyE = E + noise

	fitparams, fitcovariance = curve_fit(gauss2, noisyE, noisyI, p0 = [50, 50, 1, 1, 1, -1, 1, 4], maxfev=4000)
	plt.plot(noisyE[:,2], noisyI, 'ro', label = 'original data')
	plt.plot(E[:,2], gauss2(E, *fitparams), 'bo', label = "fit curve")
	plt.legend()
