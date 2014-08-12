#This file is for estimating the means in a sum of Gaussians as functions of certain variables. We assume the mean is a linear equation and we want the coefficients.
from __future__ import division
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import math
import random

from peak_shift_analysis import sum_gaussians_fit
from sklearn.mixture import GMM

DIR = "./xas/"

def get_snap_num(spectrum):
	return int(re.search(r'.*_([0-9]+)-.*', spectrum).group(1))

def get_energies(spectrum):
	return np.loadtxt(DIR + spectrum, usecols=(0,), dtype=float)

def get_xas_data(spectra_files):
	"""
	Parses a list of spectra files
	"""
	xas_data = [0 for _ in range(len(spectra_files))]
	for spectrum in spectra_files:
		if "Spectrum-Ave-" not in spectrum:
			intensities = np.loadtxt(DIR + spectrum, usecols=(1,), dtype=float)
			snap_num = get_snap_num(spectrum)
			xas_data[snap_num] = intensities
	if len(xas_data) > 0 and xas_data[-1] == 0:
			xas_data = xas_data[:-1]

	energies = get_energies(spectra_files[0])
	xas_data = np.array(xas_data)	
	return energies, xas_data

def get_gauss_params(energies, spectra):
	"""
	Calls the code in peak_shift_analysis.py, which fits a spectrum using a sum of Gaussians model.
	"""
	max_num, amps, means, sigmas = 0, [], [], []
	for spectrum in spectra:
		num, amps_i, means_i, sigmas_i = sum_gaussians_fit(energies, spectrum)
		amps.append(amps_i)
		means.append(means_i)
		sigmas.append(sigmas_i)
		if num > max_num:
			max_num = num
	return max_num, np.array(amps), np.array(means), np.array(sigmas)

def separate_peaks(cluster_points, num_clusters, dim=1):
	"""
	Probably will use GMM clustering
	"""
	if dim != 1 and dim != 2:
		raise ValueError("dim can only be 1 or 2")

	g = GMM(num_clusters, thresh=.0001, min_covar=.0001, n_iter=2000)
	if dim == 1:
		g.fit(cluster_points[:, 0])
	else:
		g.fit(cluster_points)
	print g.means_, g.weights_

def main():
	"""
	Gets a spectrum, finds it means, amplitudes, and sigmas from peak_shift_analysis, which fits the spectra using a sum of gaussians, and then continues this for all spectra. We have 4 pieces of information to work with: Spectra (snapshot) # and amplitudes, sigmas, and energy positions of the means in the sum of Gaussians.
	"""
	spectra_files = os.listdir(DIR)
	energies, spectra = get_xas_data(spectra_files)
	num, amps, means, sigmas = get_gauss_params(energies, spectra)
	cluster_points = to_cluster_space(means)
	separate_peaks(cluster_points, num)
	graph(cluster_points)


#EVERYTHING RELATED TO SIMULATED ANNEALING FOLLOWS


def linreg(Y, X):
	"""
	Expressed Y as a lin comb of X
	INPUT:
		Y -- each column is a different peak
		X -- each column is a different variable that may contribute to a mean/peak position
	OUTPUT:
		error -- RMS of residuals of the multiple linear regression
		coeffs -- The lin comb coefficients for Y = func(X)
	"""
	assert Y.ndim==2, "Y's dimension must be 2"
	assert X.ndim==2, "X's dimension must be 2"
	assert Y.shape[0]==X.shape[0], "X and Y must have the same number of rows"

	coeffs, residuals, rank, singular_vals = np.linalg.lstsq(X, Y)
	error = np.sqrt(np.sum(np.power(residuals, 2)) / len(residuals))
	return error, coeffs

def localswap(means):
	"""
	Picks a random row and swaps two random items OR picks an item from a row and moves it to another column
	INPUT:
		means -- Each column is a different peak/mean. Each row is a sample/instance.
	OUTPUT:
		means_cpy -- A swap or shift has been performed on means and returned as means_cpy
	"""
	assert means.ndim==2, "means must have a dimension of 2"

	means_cpy = np.copy(means)
	col_a, col_b = np.random.choice([i for i in range(len(means[0]))], 2, replace=False)
	row = random.randint(0, len(means)-1)
	means_cpy[row][col_a] = means[row][col_b]
	means_cpy[row][col_b] = means[row][col_a]
	return means_cpy

def jumble(means):
	"""
	Removes the identity of peaks (FOR TESTING ONLY)
	INPUT:
		means -- Each column is a different peak and each row is an instance/sample.
	OUTPUT:
		1 column vector of means
	"""
	assert means.ndim==2, "Dimension of means must be 2"

	len_means = sum([len(row) for row in means])
	return np.reshape(means, (len_means, 1)) #Does not edit means

def temperature(T0, iter_num):
	"""
	Returns temperature based on the iteration number
	INPUT:
		T0 -- initial temperature of system
		iter_num -- iteration number of Simulated Annealing algorithm
	OUTPUT:
		temperature
	"""
	return T0 * (.95 ** iter_num)

tabu = {}
def prob_tabu(curr_err, next_err, temperature, next_state):
	"""
	Returns acceptance probability. 1 if error decreases, less if next error is more
	INPUT:
		curr_err -- The current state's error in the Simulated Annealing
		next-err -- The next potential state's error
		temperature -- temperature of the Simulated Annealing
	OUTPUT:
	acceptance probability
	"""
	next_hash = hash(str(next_state))
	if next_hash in tabu and next_state in tabu[next_hash]:
		return 0
	elif next_hash in tabu:
		tabu[next_hash].append(0, next_state)
	else:
		tabu[next_hash] = [next_state]
	
	if next_err < curr_err:
		return 1
	return 1 / np.exp((next_err - curr_err) / temperature)

def prob(curr_err, next_err, temperature):
	"""
	Returns acceptance probability. 1 if error decreases, less if next error is more
	INPUT:
		curr_err -- The current state's error in the Simulated Annealing
		next-err -- The next potential state's error
		temperature -- temperature of the Simulated Annealing
	OUTPUT:
		acceptance probability
	"""
	if next_err < curr_err:
		return 1
	return 1 / np.exp((next_err - curr_err) / temperature)

def SA(peaks, x):
	"""
	Simulated Annealing: Used to find the labeling of peaks that minimizes the linear regression error when peaks are expressed as linear combinations of certain variables.
	INPUT:
		peaks -- Each column is a different peak
		x -- Each column is a different variables
	OUTPUT:
		best_sol -- The optimal peak labeling. Each column is a different peak.
		best_err -- The optimal error
		best_params -- The optimal linear regression coefficients for expressing means as a function of x
	"""
	assert peaks.ndim==2, "Dimension of peaks must be 2"
	assert peaks.shape[0]==x.shape[0], "Number of rows of x and peaks must match"
	assert x.ndim==2, "x should have a dimension of 2"
	
	iters, T0 = 100000, 10000
	best_sol = peaks
	best_err, best_params = linreg(best_sol, x)
	current_sol, current_err, current_params = peaks, best_err, best_params
	for i in range(1, iters+1):
		next_sol = localswap(current_sol)
		next_err, next_params = linreg(next_sol, x)
		temp = temperature(T0, i)
		if prob(current_err, next_err, temp) > random.random():
			current_sol = next_sol
			current_err = next_err
			current_params = next_params
		if next_err < best_err:
			best_sol = next_sol
			best_err = next_err
			best_params = next_params
	return best_sol, best_err, best_params

#TESTING METHODS FOLLOW
def noise(means):
	return (random.random() - 1) * .1 * means

def graph(means, x):
	for col in range(len(means[0])):
		plt.plot(x, means[:, col], 'ro')
	plt.show()

def test(means, *x_s):
	x1 = x_s[0]
	ones = np.reshape(np.ones(len(x1)), (len(x1), 1))
	x = np.hstack((np.hstack(x_s), ones))
	means += noise(means)
	means2 = jumble(means)
	final, error, params = SA(means2, x)
	return final, error, params

def t1():
	#2 separate lines, 1 variable
	x1 = np.reshape(np.random.normal(loc=.75, scale=.2, size=1000), (1000, 1))
	a1, b1 = 1, 5
	a2, b2 = 1, -5
	means = np.hstack((a1*x1 + b1, a2*x1 + b2))
	#graph(means, x1)
	print test(means, x1)

def t2():
	#2 intersecting lines, 1 variable
	x1 = np.reshape(np.random.normal(loc=.75, scale=.2, size=1000), (1000, 1))
	a1, b1 = 15, -20
	a2, b2 = -15, 5
	means = np.hstack((a1*x1 + b1, a2*x1 + b2))
	#graph(means, x1)
	print test(means, x1)

def t3():
	#2 intersecting, 1 separate lines; 1 variable
	x1 = np.reshape(np.random.normal(loc=.75, scale=.2, size=1000), (1000, 1))
	a1, b1 = 15, -20
	a2, b2 = -15, 5
	a3, b3 = 3, 5
	means = np.hstack((a1*x1 + b1, a2*x1 + b2, a3*x1 + b3))
	#graph(means, x1)
	print test(means, x1)
	
def t4():
	#3 intersecting lines, 1 variable
	x1 = np.reshape(np.random.normal(loc=.75, scale=.2, size=1000), (1000, 1))
	a1, b1 = 15, -20
	a2, b2 = -15, 5
	a3, b3 = -4, -2
	means = np.hstack((a1*x1 + b1, a2*x1 + b2, a3*x1 + b3))
	#graph(means, x1)
	print test(means, x1)

def t5():
	#3 intersecting lines (5D), 5 variables
	x1 = np.reshape(np.random.normal(loc=.75, scale=.2, size=1000), (1000, 1))
	x2 = np.reshape(np.random.normal(loc=1.5, scale=.4, size=1000), (1000, 1))
	x3 = np.reshape(np.random.normal(loc=5, scale=1, size=1000), (1000, 1))
	x4 = np.reshape(np.random.normal(loc=3.5, scale=.5, size=1000), (1000, 1))
	x5 = np.reshape(np.random.normal(loc=1, scale=.25, size=1000), (1000, 1))
	
	a1, b1, c1, d1, e1, f1 =  15,  5, 5,   8.5, 30, -20
	a2, b2, c2, d2, e2, f2 = -15, -7, 0.5, 1,    1,   5
	a3, b3, c3, d3, e3, f3 =  -4,  0, 1,   1,   10,  -2
	
	mean1 = a1*x1 + b1*x2 + c1*x3 + d1*x4 + e1*x5 + f1
	mean2 = a2*x1 + b2*x2 + c2*x3 + d2*x4 + e2*x5 + f2
	mean3 = a3*x1 + b3*x2 + c3*x3 + d3*x4 + e3*x5 + f3
	means = np.hstack((mean1, mean2, mean3))
	print test(means, x1, x2, x3, x4, x5)

def t6():
	#3 intersecting lines (1D), 5 variables
	x1 = np.reshape(np.random.normal(loc=.75, scale=.2, size=1000), (1000, 1))
	x2 = np.reshape(np.random.normal(loc=1.5, scale=.4, size=1000), (1000, 1))
	x3 = np.reshape(np.random.normal(loc=5, scale=1, size=1000), (1000, 1))
	x4 = np.reshape(np.random.normal(loc=3.5, scale=.5, size=1000), (1000, 1))
	x5 = np.reshape(np.random.normal(loc=1, scale=.25, size=1000), (1000, 1))
	
	a1, b1, c1, d1, e1, f1 =  15, 1, 1, 1, 1, -20
	a2, b2, c2, d2, e2, f2 = -15, 1, 1, 1, 1, 5
	a3, b3, c3, d3, e3, f3 =  -4, 1, 1, 1, 1, -2
	
	mean1 = a1*x1 + b1*x2 + c1*x3 + d1*x4 + e1*x5 + f1
	mean2 = a2*x1 + b2*x2 + c2*x3 + d2*x4 + e2*x5 + f2
	mean3 = a3*x1 + b3*x2 + c3*x3 + d3*x4 + e3*x5 + f3
	means = np.hstack((mean1, mean2, mean3))
	print test(means, x1, x2, x3, x4, x5)
