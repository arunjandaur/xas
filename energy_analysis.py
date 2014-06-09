from __future__ import division
import numpy as np
import scipy as sp
from scipy.signal import argrelextrema

from analysis import remove_intens_dict
from Peak import Peak

def get_extrema(energies, intensities):
	"""
	INPUT:
		energies -- A 2d numpy array. Each row is an instance (for now, that means a snapshot). Each column is an energy value
		intensities -- Same format, but intensity values instead. Indices are consistent. (i, j)th intensity corresponds to (i, j)th energy
	OUTPUT:
		ret_val -- 4 element tuple
		ret_val[0] -- minima energy values. Same format as input
		ret_val[1, 2, 3] -- maxima energies, minima intensities, and maxima intensities, respectively
	PURPOSE:
		Get all relative extrema in the energy vs. intensity spectra
	"""
	minimaE = []
	maximaE = []
	minimaI = []
	maximaI = []
	for i in range(len(intensities)):
		Irow = intensities[i]
		Erow = energies[i]
		min_indices = argrelextrema(Irow, np.less)[0]
		row_minimaE = [Erow[index] for index in min_indices]
		minimaE.append(row_minimaE)
		row_minimaI = [Irow[index] for index in min_indices]
		minimaI.append(row_minimaI)

		max_indices = argrelextrema(Irow, np.greater)[0]
		row_maximaE = [Erow[index] for index in max_indices]
		maximaE.append(row_maximaE)
		row_maximaI = [Irow[index] for index in max_indices]
		maximaI.append(row_maximaI)
	minimaE = np.array(minimaE)
	maximaE = np.array(maximaE)
	minimaI = np.array(minimaI)
	maximaI = np.array(maximaI)
	return (minimaE, maximaE, minimaI, maximaI)

def extrema_to_peaks(minimaE, maximaE, minimaI, maximaI):
	"""
	INPUT:
		minimaE -- A 2d numpy array. Each row is an instance (currently, that means a single snapshot). Each column is the energy value of a relative minima for that instance
		maximaE, minimaI, maximaI -- Same format. I indicates relative intensity maxima or minima
	OUTPUT:
		peaks -- A 2d numpy array. Each row is an instance. Each column is a Peak instance. Peaks are defined by the energy, intensity values at the maxima and the energies of the left and right minimum
	PURPOSE:
		Converting extrema of energies and intensities into peaks for the peak_tracking(*params) method to use
	IMPORTANT POINTS:
		I am assuming that the first and last extrema are minima. If not, indexing the minima based on j will be off.
	"""
	peaks = []
	for i in range(len(maximaE)):
		max_row_E = maximaE[i]
		min_row_E = minimaE[i]
		max_row_I = maximaI[i]
		row_peaks = []
		for j in range(len(max_row_E)):
			curr_max_E = max_row_E[j]
			curr_max_I = max_row_I[j]
			left_min_E = min_row_E[j]
			right_min_E = min_row_E[j+1]
			row_peaks.append(Peak(curr_max_E, curr_max_I, left_min_E, right_min_E))
		peaks.append(row_peaks)
	return peaks

def shift_cost(avg_peak, atom_peak):
	"""
	INPUT:
		avg_peak -- A single Peak from the average intensity spectra
		atom_peak -- A single Peak from an individual atom's intensity spectra
	OUTPUT:
		ret_val -- Represented as a float. The integer number of peak widths that avg_peak needs to shift to reach atom_peak
	PURPOSE:
		To calculate the cost of shifting and penalizing larger shifts. To be used with peak_tracking(*params)
	ALGORITHM:
		Find energy shift distance and divide by peak width of avg_peak. Round to get integer number of shifts. Shifts of less that one peak away shouldn't get penalized. A full peak shift (1 peak width) should be peanlized just as much as a deletion (hence, this method and delete_avg_cost(*params) both return 1. Larger shifts are penalized more.
	IMPORTANT POINTS:
		I haven't thought about whether it matters that the peak width can grow or shrink as it shifts
		Careful of first peak. Left min may not be start of peak
		Maybe intensity height differences should factor into the cost? Maybe have like a ratio between energy shift (horizontal) and intensity shift (vertical)?
	"""
	avg_E = avg_peak.getPeakEnergy()
	atom_E = atom_peak.getPeakEnergy()
	width = avg_peak.getRightEnergy() - avg_peak.getLeftEnergy()
	ret_val = round(abs(avg_E - atom_E) / width) #Integer number of peak widths that the peak is shifted by.
	return ret_val

def delete_avg_cost(avg_peak):
	"""
	INPUT:
		avg_peak -- A single Peak instance of the average intensity spectra
	OUTPUT:
		ret_val -- cost of deleting an average spectra peak
	"""
	return 1

def delete_atom_cost(atom_peak):
	"""
	INPUT:
		avg_peak -- A single Peak instance of an individual atom's intensity spectra
	OUTPUT:
		ret_val -- cost of deleting an atom's peak
	"""
	return 1

memo = {}
def peak_tracking(avg_peaks, atom_peaks):
	"""
	INPUT:
		avg_peaks -- array of Peak instances for the average intensity spectra
		atom_peaks -- array of Peak instances for an individual atom's intensity spectra
	OUTPUT:
		ret_val -- A tuple of two values.
			ret_val[0] is the diff cost of the two lists of peaks
			ret_val[1] is a list of edits (as strings) to transform the two lists together
	PURPOSE:
		Find the diff, or edit distance, of two lists of peaks
	ALGORITHM:
		Dynamic programming algorithm inspired by edit distance between two strings. Peaks are analagous to characters and the array of peaks is analogous to a string. Added features include returning both the cost and sequence of edits. Cost heuristics are above. In other words, this is a fancy diff.
	IMPORTANT POINTS:
		This algorithm starts by comparing the last peaks first. If the ends are noisy, this may misalign the edit distance for the major peaks that occur earlier on. Potential solutions: Filter the noisy peaks at the end OR flip the arrays so the first peaks are last, which means they will be processed first.
	"""

	i = len(avg_peaks)-1
	j = len(atom_peaks)-1
	
	#BASE CASES
	if i+1 == 0 and j+1 == 0:
		return (0, [])
	elif j+1 == 0:
		cost = sum([delete_avg_cost(avg_peak) for avg_peak in avg_peaks])
		instructions = ["delete AVG[{0}]".format(i) for i in range(len(avg_peaks))]
		return (cost, instructions)
	elif i+1 == 0:
		cost = sum([delete_atom_cost(atom_peak) for atom_peak in atom_peaks])
                instructions = ["delete ATOM[{0}]".format(i) for i in range(len(atom_peaks))]
                return (cost, instructions)
	#BASE CASES

	#MEMOIZATION + COMPUTING SUBPROBLEMS
	if (i-1, j-1) in memo:
		shift = memo[(i-1, j-1)]
	else:
		shift = peak_tracking(avg_peaks[:-1], atom_peaks[:-1])
		memo[(i-1, j-1)] = shift
	if (i-1, j) in memo:
		fusion = memo[(i-1, j)]
	else:
		fusion = peak_tracking(avg_peaks[:-1], atom_peaks)
		memo[(i-1, j)] = fusion
	if (i, j-1) in memo:
		split = memo[(i, j-1)]
	else:
		split = peak_tracking(avg_peaks, atom_peaks[:-1])
		memo[(i, j-1)] = split
	#MEMOIZATION + COMPUTING SUBPROBLEMS

	#TOTAL COST
	single_shift_cost = shift_cost(avg_peaks[-1], atom_peaks[-1]) #I need to use this later. That's why it gets its own variable.
	total_shift_cost = shift[0] + single_shift_cost #[-1] gets last item
	total_delete_avg_cost = fusion[0] + delete_avg_cost(avg_peaks[-1])
	total_delete_atom_cost = split[0] + delete_atom_cost(atom_peaks[-1])
	min_cost = min(total_shift_cost, total_delete_avg_cost, total_delete_atom_cost)
	#TOTAL COST
	
	#RETURN OPTIMAL COST AND INSTRUCTIONS
	if min_cost == total_shift_cost:
		ret_val = (total_shift_cost, shift[1] + ["shift AVG[{0}] --> ATOM[{1}], {2}".format(i, j, single_shift_cost)])

	elif min_cost == total_delete_avg_cost:
		ret_val = (total_delete_avg_cost, fusion[1] + ["delete AVG[{0}]".format(i)])

	elif min_cost == total_delete_atom_cost:
		ret_val = (total_delete_atom_cost, split[1] + ["delete ATOM[{0}]".format(j)])
	#RETURN OPTIMAL COST AND INSTRUCTIONS
	return ret_val

if __name__ == '__main__':
	#THIS METHOD IS FOR TESTING	

	#TEST 0
	#energies = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
	#intensities = np.array([[2, 1, 5, 10, 5, 8, 5, 1, 2, 3], [3, 2, 1, 5, 10, 5, 8, 5, 1, 2]])
	#minimaE, maximaE, minimaI, maximaI = get_extrema(energies, intensities)
	#peaks = extrema_to_peaks(minimaE, maximaE, minimaI, maximaI)
	
	#Test 1
	avg_peaks = [Peak(1, 10, 0, 2), Peak(5, 10, 4, 6)]
	atom_peaks = [Peak(1, 10, 0, 2), Peak(3, 10, 2, 4), Peak(5, 10, 4, 6)]
	
	#Test 2
	avg_peaks = [Peak(1, 10, 0, 2), Peak(3, 10, 2, 4)]
	atom_peaks = [Peak(1, 10, 0, 2), Peak(3, 10, 2, 4), Peak(5, 10, 4, 6)]
	
	#Test 3
	avg_peaks = [Peak(1, 10, 0, 2), Peak(4, 10, 3, 5)]
	atom_peaks = [Peak(1, 10, 0, 2), Peak(3, 10, 2, 4), Peak(5, 10, 4, 6)]
	
	peaks = np.array([avg_peaks, atom_peaks])
	edit_path = peak_tracking(peaks[0], peaks[1])
	print edit_path
