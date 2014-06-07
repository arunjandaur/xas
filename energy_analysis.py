import numpy as np
import scipy as sp
from scipy.signal import argrelextrema

from analysis import remove_intens_dict
from Peak import Peak

def get_extrema(energies, intensities):
	minimaE = []
	maximaE = []
	minimaI = []
	maximaI = []
	for row in intensities:
		min_indices = argrelextrema(row, np.less)[0]
		row_minimaE = [energies[index] for index in min_indices]
		minimaE.append(row_minimaE)
		row_minimaI = [row[index] for index in min_indices]
		minimaI.append(row_minimaI)

		max_indices = argrelextrema(row, np.greater)[0]
		row_maximaE = [energies[index] for index in max_indices]
		maximaE.append(row_maximaE)
		row_maximaI = [row[index] for index in max_indices]
		maximaI.append(row_maximaI)
	minimaE = np.array(minimaE)
	maximaE = np.array(maximaE)
	minimaI = np.array(minimaI)
	maximaI = np.array(maximaI)
	return (minimaE, maximaE, minimaI, maximaI)

def extrema_to_peaks(minimaE, maximaE, minimaI, maximaI):
	

def shift_cost(avg_peak, atom_peak):
	

def delete_avg_cost(avg_peak):
	

def delete_atom_cost(atom_peak):
	

memo = {}
def peak_tracking(avg_peaks, atom_peaks):
	i = len(avg_peaks)-1
	j = len(atom_peaks)-1
	
	#BASE CASES
	if i == 0 and j == 0:
		return (0, [])
	elif j == 0:
		cost = sum([delete_avg_cost(avg_peak) for avg_peak in avg_peaks])
		instructions = ["delete AVG[{0}]".format(i) for i in range(len(avg_peaks))]
		return (cost, instructions)
	elif i == 0:
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
	total_shift_cost = shift[0] + shift_cost(avg_peaks[-1], atom_peaks[-1])
	total_delete_avg_cost = fusion[0] + delete_avg_cost(avg_peaks[-1])
	total_delete_atom_cost = split[0] + delete_atom_cost(atom_peaks[-1])
	min_cost = min(total_shift_cost, total_delete_avg_cost, total_delete_atom_cost)
	#TOTAL COST
	
	#RETURN OPTIMAL COST AND INSTRUCTIONS
	if min_cost == total_shift_cost:
		ret_val = (total_shift_cost, shift[1] + ["shift AVG[{0}] --> ATOM[{1}]".format(i, j)])

	else if min_cost == total_delete_avg_cost:
		ret_val = (total_delete_avg_cost, fusion[1] + ["delete AVG[{0}]".format(i)])

	else if min_cost == total_delete_atom_cost:
		ret_val = (total_delete_atom_cost, split[1] + ["delete ATOM[{0}]".format(j)])
	#RETURN OPTIMAL COST AND INSTRUCTIONS
	return ret_val

def energy_tracking(intensities, energies, E_peak):
        """
        INPUT: Matrix with each row a different atom. 1000 columns, each an intensity corresponding to a different energy.
        OUTPUT: Matrix with each row a different atom. A few columns. Each column value contains an atom's col-th energy peak.
        PURPOSE: Energies are not independent. In other words, atom i in snapshot j has a peak at energy e but the same atom i in snaphot j' has its corresponding peak at e'. As a result, correlating coordinate instances to intensities at a particular energy is nonsensical because the energies at which a phenomenon happens with respect to the coordinates shift.
        HOW: Implemented by finding local maxima.
        INVARIANTS:
                All energies are equally spaced. If not, alter the signal.argrelextrema line.
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
                if len(maxima) == 0:
                        max_inten = max(I_row)
                        index = list(I_row).index(max_inten)
                        maxima = [E_row[index]]
                nearest = maxima[0]
                nearest_dist = abs(E_peak-nearest)
                for maximum in maxima:
                        if abs(E_peak-maximum) < nearest_dist:
                                nearest = maximum
                                nearest_dist = abs(E_peak-maximum)
                new_energies.append([nearest])
                nearest_index = list(E_row).index(nearest)
                new_intens.append([I_row[nearest_index]])
                i += 1
        intensities = np.array(new_intens)
        energies = np.array(new_energies)
        return intensities, energies
