from __future__ import division
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from math import *
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter1d

def gauss(E, sigma, a, b, c):
	x = E[:, 0]
	x2 = E[:, 1]
	energy = E[:, 2]
	A = 1 / (sigma * sqrt(2*pi))
	return A * np.exp(-.5 * np.power((energy - (a*x+b*x2+c)) / sigma, 2))

def gauss2(E, sigma1, sigma2, a1, b1, c1, a2, b2, c2):
	return gauss(E, sigma1, a1, b1, c1) + gauss(E, sigma2, a2, b2, c2)

def gauss_simple(E, A, avg, sigma):
	return A * np.exp(-.5 * np.power((E-avg) / sigma, 2))

def num_gaussians(data):
	newdata = gaussian_filter1d(data, sigma=.1, mode='constant')
	return newdata

if __name__ == "__main__":
	E = np.linspace(0, 8, 100)
	I = gauss_simple(E, 8, 5, 1) + gauss_simple(E, 8, 6, 1)
	newI = num_gaussians(I)
	plt.plot(E, newI, 'ro', label='fit')
	plt.plot(E, I, 'b', label='original')
	plt.show()
	"""	
	X = (np.random.normal(loc=1.16, scale=.16, size=10) - 1.16) / .16
	X2 = (np.random.normal(loc=3, scale=.3, size=10) - 3) / .3
	sigma1 = .5
	sigma2 = .5
	a1 = .8
	b1 = 0
	c1 = 3
	a2 = -.8
	b2 = 0
	c2 = 3
	E = np.array([[], [], []])

	for i in range(len(X)):
		x = X[i]
		x2 = X2[i]
		mean = a1*x + b1*x2 + c1
		mean2 = a2*x + b2*x2 + c2
		xdata = np.linspace(mean-2.5*sigma1, mean2+2.5*sigma2, 1000)
		x_s = [x for _ in range(len(xdata))]
		x2_s = [x2 for _ in range(len(xdata))]
		temp = np.vstack((x_s, x2_s, xdata))
		E = np.hstack((E, temp))

	E = np.transpose(E)
	I = gauss2(E, sigma1, sigma2, a1, b1, c1, a2, b2, c2)
	noise = random.random(len(I)) * np.mean(I)*.05
	noisyI = I + noise
	noise = np.transpose(np.vstack((np.zeros(len(E)), np.zeros(len(E)), random.random(len(E)) * .05*np.mean(E[:,2]))))
	noisyE = E + noise

	fitparams, fitcovariance = curve_fit(gauss2, noisyE, noisyI, p0 = [50, 50, 1, 1, 1, 1, 1, 1], maxfev=4000)
	plt.plot(noisyE[:,2], noisyI, 'ro', label = 'original data')
	plt.plot(E[:,2], gauss2(E, *fitparams), 'bo', label = "fit curve")
	plt.legend()
	"""
