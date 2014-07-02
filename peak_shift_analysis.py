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

def derivative(data, interval, order):
	#TODO: index out of bounds error checking
	retval = data
	for _ in range(order):
		retval = (retval[:, 1:retval.shape[1]-1] - retval[:, 0:retval.shape[1]-2]) / interval
	return retval

def num_gaussians(data, interval):
	deriv2 = derivative(data, interval, 2)
	left = deriv2[:, 0:deriv2.shape[1]-3]
	right = deriv2[:, 1:deriv2.shape[1]-2]
	right2 = deriv2[:, 2:deriv2.shape[1]-1]

	nums = []
	for i in range(len(left)):
		num = 0
		for j in range(len(left[0])):
			left_val = left[i][j]
			right_val = right[i][j]
			right2_val = right2[i][j]
			if right_val == 0 and ((left_val < 0 and right2_val > 0) or (left_val > 0 and rightval < 0)):
				num += 1
			elif (left_val < 0 and right_val > 0) or (left_val > 0 and right_val < 0):
				num += 1
		nums.append(num)
	nums = np.array(nums)
	nums = np.ceil(nums / 2.0)
	return nums

def get_zero_crossings(energies, data, interval):
	deriv2 = derivative(data, interval, 2)
	left = deriv2[:, 0:deriv2.shape[1]-3]
	right = deriv2[:, 1:deriv2.shape[1]-2]
	right2 = deriv2[:, 2:deriv2.shape[1]-1]

	zero_crossings = []
        for i in range(len(left)):
                row_zeros = []
                for j in range(len(left[0])):
                        left_val = left[i][j]
                        right_val = right[i][j]
                        right2_val = right2[i][j]
                        if right_val == 0 and ((left_val < 0 and right2_val > 0) or (left_val > 0 and rightval < 0)):
				row_zeros.append(energies[j+1])
                        elif (left_val < 0 and right_val > 0) or (left_val > 0 and right_val < 0):
				row_zeros.append(energies[j])
                zero_crossings.append(row_zeros)
        zero_crossings = np.array(zero_crossings)
        return zero_crossings

def to_arc_space(zeros, sigmas):
	arc_data = []
	for i in range(len(sigmas)):
		sigma = sigmas[i]
		for zero in zeros[i]:
			arc_data.append([zero, sigma])
	return np.array(arc_data)

def gauss_simple(E, A, avg, sigma):
	return A * np.exp(-.5 * np.power((E-avg) / sigma, 2))

def smooth_gaussians(data, sigmas):
	retval = np.empty(data.shape)
	for sig in sigmas:
		newdata = np.empty(data.shape)
		gaussian_filter1d(data, sigma=sig, output=newdata, mode='constant', cval=0.0)
		retval = np.vstack((retval, newdata))
	return retval

if __name__ == "__main__":
	E = np.linspace(0, 8, 100)
	I = gauss_simple(E, .5, 3, .5) + gauss_simple(E, 8, 7, .5) + gauss_simple(E, 1, 5, .01)
	sigmas = np.linspace(1, 10, 10)
	smoothed = smooth_gaussians(I, sigmas)
	zero_crossings = get_zero_crossings(E, smoothed, E[1]-E[0])
	arc_data = to_arc_space(zero_crossings, sigmas)
	
	plt.plot(arc_data[:, 0], arc_data[:, 1], 'ro', label = 'arc space')
	plt.show()
	"""
	for i in range(len(smoothed)):
		plt.plot(E, smoothed[i], 'b', label='fit')
	plt.plot(E, I, 'ro', label='original')
	plt.show()
	"""
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
