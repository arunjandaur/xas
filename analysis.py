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
        for row in intensities:
                energies = row[0].keys()
                energies = sorted([float(energy) for energy in energies])
                energies = [str(energy) for energy in energies]
                new_row = []
                for energy in energies:
                        inten = row[0][energy]
                        new_row.append(inten)
                new_intens.append(new_row)
        intensities = np.array(new_intens)
	return intensities

def energy_tracking(intensities):
	"""
	INPUT: Matrix with each row a different atom. One column. Contains a dictionary of 1000 (energy, intensity) pairs.
	OUTPUT: Matrix with each row a different atom. A few columns. Each column value contains an atom's col-th energy peak.
	PURPOSE: Energies are not independent. In other words, atom i in snapshot j has a peak at energy e but the same atom i in snaphot j' has its corresponding peak at e'. As a result, correlating coordinate instances to intensities at a particular energy is nonsensical because the energies at which a phenomenon happens with respect to the coordinates shift.
	HOW: Implemented by finding local maxima.
	INVARIANTS:
		All energies are equally spaced. If not, alter the singal.argrelextrema line.
	"""
	intensities = _remove_intens_dict(intensities)
	new_intens = []
	for i in range(len(intensities)):
		row = intensities[i]
		#maxima_indices = signal.argrelextrema(row, np.greater)[0] #For minima, do np.less
		widths = np.array([.5, .7, .9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1])
		maxima_indices = signal.find_peaks_cwt(row, widths)
		maxima = [row[maxima_index] for maxima_index in maxima_indices] #Replace row with intensities that correspond to indices returned.
		new_intens.append(maxima[0:4])
		print len(maxima)
		i += 1
	intensities = np.array(new_intens)
	return intensities

def correlations(data, intens):
	data2 = np.transpose(data)
	intens2 = np.transpose(intens)
	for row in data2:
		print "row"
		for i in range(343, 362):
			coeff = stats.pearsonr(row, intens2[i])[0]
			print "Pearson coefficient: ", coeff

def lin_reg(coords, intensities):
	intensities = energy_tracking(intensities)
	ones_column = np.array([ones(len(coords))]).T
	coords = normalize(coords)
	#correlations(coords, intensities)
	coords = np.hstack((coords, ones_column)) #matrix variable instantiations to be right multiplied by coeff matrix to obtain intensities. Goal is to find coeff matrix.
	coeffs = linalg.lstsq(coords, intensities)[0] #1st elem of tuple. In each column (corresponding to an energy), each row represents a coefficient to fit the coordinates to the intensity at a certain energy. The last row is the constant b.
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
