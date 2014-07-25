#This file is for estimating the means in a sum of Gaussians as functions of certain variables. We assume the mean is a linear equation and we want the coefficients.

import os
import re

DIR = "./xas/"

def get_snap_num(spectrum):
	return int(re.search(r'.*_([0-9]+)-.*', spectrum).group(1))

def get_energies(spectrum):
	return np.loadtxt(spectrum, usecols=(0,))

def get_xas_data(spectra_files):
	"""
	Parses a list of spectra files
	"""
	xas_data = [0 for _ in range(len(spectra_files))]
        for spectrum in spectra_files:
                if "Spectrum-Ave-" not in spectrum:
                        intensities = np.loadtxt(spectrum, usecols=(1,))
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
	amps, means, sigmas = [], [], []
	for spectrum in spectra:
		params = sum_gaussians_fit(energies, spectrum)
		amps.append(params[0])
		means.append(params[1])
		sigmas.append(params[2])
	return np.array(amps), np.array(means), np.array(sigmas)

def to_cluster_space(gauss_params):
	"""
	gauss_params could be one of three: amps, means, or sigmas. For now, we will use this method to plot and cluster means vs snapshot numbers. Later we may want to track amplitudes and express those as lin. combs. of external variables (hence the generality). The row index of gauss_params indicates the snapshot number at which the data at that row is found.
	"""
	cluster_points = [] #Points that will be clustered later
	for snap_num in range(len(gauss_params)):
		row_of_gaussians = gauss_params[snap_num]
		for gauss_param in row_of_gaussians:
			cluster_points.append([gauss_param, snap_num])
	return np.array(cluster_points)

def separate_peaks():
	"""
	Probably will use GMM clustering
	"""
	pass

def main():
	"""
	Gets a spectrum, finds it means, amplitudes, and sigmas from peak_shift_analysis, which fits the spectra using a sum of gaussians, and then continues this for all spectra. We have 4 pieces of information to work with: Spectra (snapshot) # and amplitudes, sigmas, and energy positions of the means in the sum of Gaussians.
	"""
	spectra_files = os.listdir(DIR)
	energies, spectra = get_xas_data(spectra_files)
	amps, means, sigmas = get_gauss_params(energies, spectra)
	cluster_points = to_cluster_space(means)
