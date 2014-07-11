from __future__ import division
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from math import *
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter1d

NUM = 3 #Max number of gaussians
VARS = 2 #Max number of variables

def gauss_creator(num_of_gauss):
	"""
        Higher order function that returns a gaussian curve function 
        with the number of gaussians specified
        The variable inputs will be of the form
        x_position, amplitude1, mean1, sigma1, amplitude2, mean2, sigma2, etc...
        """
        if num_of_gauss <= 0:
                raise Exception("gauss_creator needs a nonzero positive input")
        def make_func(func):
                return lambda E, A, avg, sigma, *args: func(E, *args) + A * np.exp(-.5 * np.power((E-avg)/sigma, 2))
        func = lambda E, A, avg, sigma : A * np.exp(-.5 * np.power((E-avg) / sigma, 2))
        for _ in range(num_of_gauss-1):
                func = make_func(func)
        return func

def gauss(E, A, sigma, a, b):
	x = E[:, 0]
	energy = E[:, 1]
	#x2 = E[:, 1]
	#energy = E[:, 2]
	return A * np.exp(-.5 * np.power((energy - (a*x+b)) / sigma, 2))

def gauss3(E, A1, sigma1, A2, sigma2, A3, sigma3, a1, b1, a2, b2, a3, b3):
	return gauss(E, A1, sigma1, a1, b1) + gauss(E, A2, sigma2, a2, b2) + gauss(E, A3, sigma3, a3, b3)

def derivative(data, interval, order):
	#TODO: index out of bounds error checking
	retval = data
	for _ in range(order):
		retval = (retval[:, 1:retval.shape[1]-1] - retval[:, 0:retval.shape[1]-2]) / interval
	return retval

def smooth_gaussians(data, sigmas):
	#TODO: Indexing error checking
	newdata = np.empty(data.shape)
        gaussian_filter1d(data, sigma=sigmas[0], order=2, output=newdata, mode='reflect')
	retval = newdata
        for sig in sigmas[1:]:
        	gaussian_filter1d(data, sigma=sig, order=2, output=newdata, mode='reflect')
        	retval = np.vstack((retval, newdata))
	return retval

def get_zero_crossings(energies, data, interval):
	#TODO: index out of bounds error checking
	deriv2 = data#derivative(data, interval, 2)
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
                        if right_val == 0 and ((left_val < 0 and right2_val > 0) or (left_val > 0 and right_val < 0)):
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

def find_pairs(pairs, crossings):
	#TODO: Fix problem when crossings is empty
	new_pairs = []
	copy = list(np.copy(crossings)) #This is acting weird
	for i in range(len(pairs)):
		pair = pairs[i]
		if len(copy) == 0:
			new_pairs.extend(pairs[i:])
			return np.array(new_pairs)
		left = pair[0]
		right = pair[1]
		min_left_dist = 40
		min_right_dist = 40
		min_left = 0
		min_right = 0
		for cross in copy:
			if abs(cross - left) < min_left_dist:
				min_left_dist = abs(cross - left)
				min_left = cross
			elif abs(cross - right) < min_right_dist:
				min_right_dist = abs(cross - right)
				min_right = cross
		copy.remove(min_left)
		copy.remove(min_right)
		if len(new_pairs) > 0:
			prev = new_pairs[i-1]
			prev_left = prev[0]
			prev_right = prev[1]
			if min_left > prev_left and min_left < prev_right and min_right > prev_left and min_right < prev_right:
				new_pairs[i-1][1] = min_left
				new_pairs.append([min_right, right])
			else:
				new_pairs.append([min_left, min_right])
		else:
			new_pairs.append([min_left, min_right])
	if len(copy) == 2:
		#GOOD
		new_pairs.append([copy[0], copy[1]])
	elif len(copy) != 0:
		return False
	return np.array(new_pairs)

def label_arches(zero_crossings):
	i = len(zero_crossings)-1
	prev_pairs = []
	while i >= 0:
		crossings = zero_crossings[i]
		prev_pairs = find_pairs(prev_pairs, crossings)
		if type(prev_pairs) == bool:
			return False
		i -= 1
	return prev_pairs

def find_arches():
	pass

def estimate_means(arches):
	means = []
	for arch in arches:
		left = arch[0]
		right = arch[1]
		mean = (left + right) / 2.0
		means.append(mean)
	return np.array(means)

def estimate_sigmas(arches):
	sigmas = []
	for arch in arches:
		left = arch[0]
		right = arch[1]
		sigma = abs(left - right) / 2.0
		sigmas.append(sigma)
	return np.array(sigmas)

def estimate_amplitudes(energies, intensities, means):
	amps = []
	for mean in means:
		min_dist = 40
		amp = 0
		for i in range(len(energies)):
			energy = energies[i]
			if abs(energy - mean) < min_dist:
				min_dist = abs(energy - mean)
				amp = intensities[i]
		amps.append(amp)
	return amps

def gauss_simple(E, A, avg, sigma):
	return A * np.exp(-.5 * np.power((E-avg) / sigma, 2))

def gauss_simple2(E, A1, avg1, sigma1, A2, avg2, sigma2, A3, avg3, sigma3):
	return gauss_simple(E, A1, avg1, sigma1) + gauss_simple(E, A2, avg2, sigma2) + gauss_simple(E, A3, avg3, sigma3)

def estimate_num_gauss(arches, tol, E, I):
	error = 1000
	n = 1
	m = len(arches)
	params = []
	while error > tol:
		if n > m:
			return m #maybe instead we should remember the n with the least error
		n_dominant = arches[0:n]
		gauss_func = gauss_creator(n)
		means = estimate_means(n_dominant)
		sigmas = estimate_sigmas(n_dominant)
		amps = estimate_amplitudes(E, I, means)
		initialparams = []
		for i in range(n):
			initialparams.append(amps[i])
			initialparams.append(means[i])
			initialparams.append(sigmas[i])
		#while len(initialparams) != 3*NUM:
			#initialparams.append(0) #A
			#initialparams.append(0) #mean
			#initialparams.append(.1) #Sigma can't be 0
		params, covar = curve_fit(gauss_func, E, I, p0=initialparams, maxfev=4000)
		error = np.sqrt(1/len(E) * np.sum(np.power(I - gauss_func(E, *params), 2)))
		#error = np.sqrt(np.mean(np.sum(np.diag(covar)))) #Should I take the mean before the sqrt?
		n += 1
	return n-1, params

def estimate_mean_coeffs(means):
	coeffs = []
	for mean in means:
		coeffs.append(0)
		coeffs.append(mean)
	return coeffs

def graph():
	plt.figure(1)
	plt.subplot(221)
        for i in range(len(smoothed)):
	        plt.plot(E[:, 1][0:1000], smoothed[i], 'b', label='fit')
        plt.plot(E[:, 1][0:1000], I[0:1000], 'ro', label='original')
        plt.subplot(222)
        plt.plot(arc_data[:, 0], arc_data[:, 1], 'go', label='arc data')
        plt.show()

if __name__ == "__main__":
	E = np.array([[], []])
	X = (np.random.normal(loc=1.16, scale=.16, size=100))
	for i in range(len(X)):
        	x = X[i]
                energies = np.linspace(0, 30, 1000)
                x_s = [x for _ in range(len(energies))]
                temp = np.vstack((x_s, energies))
                E = np.hstack((E, temp))
	E = np.transpose(E)
	I = gauss3(E, .16, .25, .16, .25, 0, .001, 0, 14.5, 0, 15.5, 0, 0)
	I_1 = I[0:1000]
	sigmas = np.linspace(.001, 500, 1000)
	smoothed = smooth_gaussians(I_1, sigmas)
	zero_crossings = get_zero_crossings(E[:, 1][0:1000], smoothed, E[:, 1][1]-E[:, 1][0])
	#print [len(cross) for cross in zero_crossings]
	arc_data = to_arc_space(zero_crossings, sigmas)
	#graph()
	arches = label_arches(zero_crossings)
	#print "Arches" + str(arches)
	
	num, params = estimate_num_gauss(arches, .001, E[:, 1][0:1000], I_1)
	newparams = []
	i = 0
	means = []
	while i < len(params)-2:
		amp = params[i]
		mean = params[i+1]
		sigma = params[i+2]
		newparams.append(amp)
		newparams.append(sigma)
		means.append(mean)
		i += 3
	newparams.extend(estimate_mean_coeffs(means))
	print params
	print newparams
	finalparams, covar = curve_fit(gauss3, E, I, p0=newparams, maxfev=4000)
	print finalparams
	
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
