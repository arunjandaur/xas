from __future__ import division
import numpy as np
from numpy import random
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

def single_peak_variable_fit():
	mean = lambda x: x
	gauss = lambda E, x: 1 / sqrt(2*pi) * np.exp(-.5 * np.power(E - mean(x), 2))

	x = 50
	interval = [mean(x)-2.0, mean(x)+2.0]
	#energies = np.linspace(interval[0], interval[1], 1000)
	dist = norm(loc=mean(x), scale=1)
	energies = dist.rvs(size=1000)
	print energies
	intens = gauss(energies, x)
	return norm.fit(energies)[0]
	#return curve_fit(gauss, energies, intens, p0=9)[0]
	#return norm.fit(np.transpose(np.vstack((energies, intens))), loc=500)[0]

if __name__ == "__main__":
	print single_peak_variable_fit()
