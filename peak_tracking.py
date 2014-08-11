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

def main2():
	x = np.random.normal(loc=.75, scale=.2, size=1000)
	a1 = 15
	b1 = -20
	a2 = -15
	b2 = 5
	means = np.vstack((a1*x + b1, a2*x + b2))
	width = .025
	widths = [width*(i+1) for i in range(9)]

	for bin_width in widths:
		hist_means, hist_freq = get_binned_hist(means, max([max(means_i) for means_i in abs(means)]), bin_width)
		hist_label = "Bin width = " + str(bin_width)
		plt.plot(hist_means, hist_freq, 'b', label=hist_label)
		plt.legend()
		plt.show()

	plt.plot(x, means[0], 'bo')
	plt.plot(x, means[1], 'go')
	plt.show()

def linreg(Y, X):
	coeffs, residuals, rank, singular_vals = np.linalg.lstsq(X, Y)
	error = np.sqrt(np.sum(np.power(residuals, 2)) / len(residuals))
	return error, coeffs

def localswap(means):
	means_cpy = np.copy(means)
	col_a, col_b = np.random.choice([i for i in range(len(means[0]))], 2, replace=False)
	row = random.randint(0, len(means)-1)
	means_cpy[row][col_a] = means[row][col_b]
	means_cpy[row][col_b] = means[row][col_a]
	return means_cpy

def jumble(means):
	means2 = np.copy(means)
	for i in range(1000000):#len(means2)):
		means2 = localswap(means2)
	return means2

def temperature(T0, iter_num):
	return T0 * (.95 ** iter_num)

def prob(curr_err, next_err, temperature):
	if next_err < curr_err:
		return 1
	return 1 / np.exp((next_err - curr_err) / temperature)

def SA(peaks, x):
	iters, T0 = 100000, 10000
	best_sol = peaks
	best_err, best_params = linreg(best_sol, x)
	current_sol, current_err, current_params = peaks, best_err, best_params
	for i in range(1, iters+1):
		next_sol = localswap(current_sol)
		next_err, next_params = linreg(next_sol, x)
		temp = temperature(T0, i)
		print next_err
		if prob(current_err, next_err, temp) > random.random():
			current_sol = next_sol
			current_err = next_err
			current_params = next_params
			#print current_err
		if next_err < best_err:
			best_sol = next_sol
			best_err = next_err
			best_params = next_params
	return best_sol, best_err, best_params

def SAtest():
	x1 = np.reshape(np.random.normal(loc=.75, scale=.2, size=1000), (1000, 1))
	x2 = np.reshape(np.random.normal(loc=1.5, scale=.4, size=1000), (1000, 1))
	x3 = np.reshape(np.random.normal(loc=5, scale=1, size=1000), (1000, 1))
	a1, b1, c1, d1 = 15, 5, 3, -20
	a2, b2, c2, d2 = -15, 2, 5, 5
	a3, b3, c3, d3 = -5, -4, 1, 2
	means = np.hstack((a1*x1 + b1*x2 + c1*x3 + d1, a2*x1 + b2*x2 + c2*x3 + d2, a3*x1 + b3*x2 + c3*x3 + d3))
	means += (random.random()-1) * .4 * means
	#plt.plot(x, means[:, 0], 'bo')
	#plt.plot(x, means[:, 1], 'ro')
	#plt.plot(x, means[:, 2], 'go')
	#plt.show()
	
	ones = np.reshape(np.ones(len(x1)), (len(x1), 1))
	x = np.hstack((x1, x2, x3, ones))
	means2 = jumble(means)
	final, error, params = SA(means2, x)
	print final
	print error
	print params
	print linreg(means, x)
