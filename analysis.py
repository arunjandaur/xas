from __future__ import division
import numpy as np
import scipy as sp
from scipy import stats
from scipy import signal
from numpy import arange, array, ones, linalg
from pylab import plot, show
from mdp.nodes import PCANode

def pretty_print():
	np.set_printoptions(threshold='nan', precision=5)

def normalize(data):
	"""
	INPUT: Matrix where each row is an atom and the columns are different coordinates. Each column is a coordinate type.
	OUTPUT: Normalized matrix in the same format
		Norm function = (X - <X>) / stdev(X)
	INVARIANTS:
		Every column must have the same length
		All elements need to be floats
	"""
	data = np.transpose(data)
	length = len(data[0]) # I expect every column of the input to the same length
	for i in range(len(data)):
		row = data[i]
		mean = sum(row) / length # <X>
		row = (row - mean) / np.std(data)
		data[i] = row
		i += 1
	return np.transpose(data)

def _remove_intens_dict(intensities):
	"""
	INPUT: Matrix with each row a different atom. One column. Contains a dictionary of 1000 (energy, intensity) pairs.
	OUTPUT: Matrix such that intensities[row] is an array of intensities.
	"""
	new_intens = []
	new_energies = []
        for row in intensities:
                energies = row[0].keys()
                energies = sorted([float(energy) for energy in energies])
                energies = [str(energy) for energy in energies]
                new_row = []
                for energy in energies:
                        inten = row[0][energy]
                        new_row.append(inten)
                new_intens.append(new_row)
		new_energies.append(energies)
        intensities = np.array(new_intens)
	energies = np.array(new_energies)
	return (intensities, energies)

def energy_tracking(intensities, energies, E_peak):
	"""
	INPUT: Matrix with each row a different atom. 1000 columns, each an intensity corresponding to a different energy.
	OUTPUT: Matrix with each row a different atom. A few columns. Each column value contains an atom's col-th energy peak.
	PURPOSE: Energies are not independent. In other words, atom i in snapshot j has a peak at energy e but the same atom i in snaphot j' has its corresponding peak at e'. As a result, correlating coordinate instances to intensities at a particular energy is nonsensical because the energies at which a phenomenon happens with respect to the coordinates shift.
	HOW: Implemented by finding local maxima.
	INVARIANTS:
		All energies are equally spaced. If not, alter the singal.argrelextrema line.
	"""
	new_intens = []
	new_energies = []
	for i in range(len(intensities)):
		I_row = intensities[i]
		E_row = energies[i]
		maxima_indices = signal.argrelextrema(I_row, np.greater)[0] #For minima, do np.less
		#widths = np.array([.1, .2, .3, .4, .5, .7, .9, 1.1, 1.3])
		#maxima_indices = signal.find_peaks_cwt(row, widths)
		maxima = [float(E_row[maxima_index]) for maxima_index in maxima_indices] #Replace row with intensities that correspond to indices returned.
		nearest = maxima[0]
		nearest_dist = abs(E_peak-nearest)
		for maximum in maxima:
			if abs(E_peak-maximum) < nearest_dist:
				nearest = maximum
				nearest_dist = abs(E_peak-maximum)
		new_energies.append([nearest])
		new_intens.append([float(I_row[np.where(E_row == str(nearest))])])
		i += 1
	intensities = np.array(new_intens)
	energies = np.array(new_energies)
	return intensities, energies

def correlations(data, intens):
	data2 = np.transpose(data)
	intens2 = np.transpose(intens)
	for row in data2:
		print "row"
		for i in range(343, 362):
			coeff = stats.pearsonr(row, intens2[i])[0]
			print "Pearson coefficient: ", coeff

def filter_intens(intensities, left, right):
	new_intens = []
	for i in range(len(intensities)):
		new_intens.append(intensities[i][left:right])
		i += 1
	return np.array(new_intens)

def filter_peak(intensities, peak_energy, interval):
	new_intens = []
	new_energies = []
	energies = intensities[0][0].keys()
	energies = sorted([float(energy) for energy in energies])
	peak_index = energies.index(peak_energy)
	left = peak_index - interval
	right = peak_index + interval
	#energies = energies[peak_energy-interval:peak_energy+interval] #todo: prevent array out of bounds exception
	intensities, energies = _remove_intens_dict(intensities)
	for i in range(len(intensities)):
		new_intens.append(intensities[i][left:right]) #same todo as above
		new_energies.append(energies[i][left:right])
	return (np.array(new_intens), np.array(new_energies))

def lin_reg(coords, intensities):
	pretty_print()
	peak_energy = 5.71
	interval = 32
	filtered_intens, filtered_energies = filter_peak(intensities, peak_energy, interval)
	tracked_intens, tracked_energies = energy_tracking(filtered_intens, filtered_energies, peak_energy)
	ones_column = np.array([ones(len(coords))]).T
	coords = normalize(coords)
	coords = np.hstack((coords, ones_column)) #matrix variable instantiations to be right multiplied by coeff matrix to obtain intensities. Goal is to find coeff matrix.
	print tracked_intens
	coeffs = linalg.lstsq(coords, tracked_intens)[0] #1st elem of tuple. In each column (corresponding to an energy), each row represents a coefficient to fit the coordinates to the intensity at a certain energy. The last row is the constant b.
	print coeffs
	
def PCA():
	col1 = np.array([[1.0, 2, 3, 4, 5, 6, 7, 8, 9]]).T
        col2 = np.array([[2.0, 4, 6, 8, 10, 12, 14, 16, 18]]).T
        matr12 = np.hstack((col1, col2))
        matr_arr = [matr12]
        pca_node = PCANode()
        d_arr = []

        for arr in matr_arr:
                result = pca_node.execute(arr)
                d_arr.append(pca_node.d)
                print pca_node.d
