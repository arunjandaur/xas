#This file is for estimating the means in a sum of Gaussians as functions of certain variables. We assume the mean is a linear equation and we want the coefficients.

import os
import re
import matplotlib.pyplot as plt
import numpy as np

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

def to_cluster_space(gauss_params):
	"""
	gauss_params could be one of three: amps, means, or sigmas. For now, we will use this method to plot and cluster means vs snapshot numbers. Later we may want to track amplitudes and express those as lin. combs. of external variables (hence the generality). The row index of gauss_params indicates the snapshot number at which the data at that row is found.
	"""
	cluster_points = [] #Points that will be clustered later
	for snap_num in range(len(gauss_params)):
		row_of_gaussians = gauss_params[snap_num]
		for gauss_param in row_of_gaussians:
			cluster_points.append([snap_num, gauss_param])
	return np.array(cluster_points)

def separate_peaks(cluster_points, num_clusters):
	"""
	Probably will use GMM clustering
	"""
	g = GMM(num_clusters, thresh=.0001, min_covar=.0001, n_iter=2000)
	g.fit(cluster_points)
	print g.means_, g.weights_

def graph(cluster_points):
	plt.plot(cluster_points[:, 0], cluster_points[:, 1], 'bo')
	plt.show()

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
