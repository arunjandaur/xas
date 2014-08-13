#This file is for estimating the means in a sum of Gaussians as functions of certain variables. We assume the mean is a linear equation and we want the coefficients.
from __future__ import division
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import math
import random

from sum_gauss_fit import sum_gaussians_fit
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

def linreg(data):
    """
    Expressed Y as a lin comb of X
    INPUT:
        data -- clusters of data
    OUTPUT:
        error -- RMS of residuals of the multiple linear regression
        coeffs -- The lin comb coefficients for Y = func(X)
    """
    assert data.ndim==2, "data's dimension must be 2"
    
    sum_res_sqr = 0
    num_points = 0
    params = np.array([],ndmin = 2).reshape(len(data[0,0][0]),0) 
    for pair in data:
        #Do regression
        X = pair[0]
        Y = pair[1]
        coeffs, res, rank, singular_vals = np.linalg.lstsq(X, Y)
        
        #Makes coeffs 2d for concatenation
        if coeffs.ndim == 1:
            coeffs.shape += (1,)
        #print coeffs 
        
        #Update Values
        sum_res_sqr += res
        num_points += len(Y)
        params = np.hstack((params, coeffs))
    
    #print params
    error = np.sqrt( sum_res_sqr/ num_points )
    return error, params

def localswap(data, num_2_swap = 1):
    """
    Picks a random row and swaps two random items OR picks an item from a row and moves it to another column
    INPUT:
        data -- clusters of data
        num_2_wap -- number of data points to swap or shift
    OUTPUT:
        data_change -- A swap or shift has been performed on data and returned
    """
    assert data.ndim==2, "data must have a dimension of 2"
    assert num_2_swap > 0, "num_2_swap must be positive"

    data_change = np.copy(data)
    for _ in range(num_2_swap):
        clus_a_i, clus_b_i = np.random.choice(range(len(data)), 2, replace=False)
        #taken from clus_a -> clus_b

        point = np.random.randint(len(data[clus_a_i,1]))
        clus_a = data[clus_a_i]
        clus_b = data[clus_b_i]
       
        data_change[clus_b_i,0] = np.append(clus_b[0], clus_a[0][[point]], axis=0)  
        data_change[clus_b_i,1] = np.append(clus_b[1], clus_a[1][point])
        data_change[clus_a_i,0] = np.delete(clus_a[0], point, axis=0)  
        data_change[clus_a_i,1] = np.delete(clus_a[1], point, axis=0)
        

    return data_change

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

def SA(x_data, y_data):
    """
    Simulated Annealing: Used to find the labeling of peaks that minimizes the linear regression error when peaks are expressed as linear combinations of certain variables.
    INPUT:
        x_data -- Independent Data, columns are variables. Each row is a snapshot.
        y_data -- Dependent Data, jagged matrix. Each row is a snapshot 
    OUTPUT:
        best_sol -- The optimal peak labeling. 
        best_err -- The optimal error
        best_params -- The optimal linear regression coefficients for expressing means as a function of x
    """
    assert x_data.ndim==2, "Dimension of x_data must be 2"
    assert x_data.shape[0]==y_data.shape[0], "Number of rows of x_data and y_data should be the same"
    
    #Flattens both input arrays
    #Done to remove jaggedness of y_data
    new_x_array = []
    new_y_array = []
    counter = 0
    for i in xrange(len(x_data)):
        for y in y_data[i]:
            counter += 1
            new_y_array.append(y)
            new_x_array.append(x_data[i])
   
    #Approximate the number of groups
    num_cluster = int(np.ceil(counter/len(y_data)))

    paired_data = np.zeros((num_cluster,2),dtype = (list,list))
    step = int(np.floor(len(new_y_array)/num_cluster))
    for i in range(num_cluster):
        paired_data[i,0] = np.array(new_x_array[i*step:(i+1)*step])
        paired_data[i,1] = np.array(new_y_array[i*step:(i+1)*step])
    if step < len(new_y_array)/num_cluster:
        paired_data[num_cluster-1,0].append(new_x_array[num_cluster*step:])
        paired_data[num_cluster-1,1].append(new_y_array[num_cluster*step:])
    
    iters, T0 = 100000, 10000
    best_sol = paired_data
    best_err, best_params = linreg(best_sol)
    current_sol, current_err, current_params = best_sol, best_err, best_params
    for i in range(1, iters+1):
        next_sol = localswap(current_sol)
        next_err, next_params = linreg(next_sol)
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
    final, error, params = SA(x, means)
    return final, error, params

def t0():
    #2 separate lines, 1 variable
    x1 = np.random.normal(loc=.75, scale=.2, size=100)
    a1, b1 = 15, -20
    a2, b2 = -15, 5
    mean1 = a1*x1 + b1
    mean2 = a2*x1 + b2
    mean2_2 = []
    x1_2 = []
    for i in range(len(x1)):
            x1_i = x1[i]
            if not (x1_i <= (.8333 + .05) and x1_i >= (.8333 - .05)) or random.random() > .65:
                    x1_2.append(x1_i)
                    mean2_2.append(mean2[i])
                    
    mean1 = np.reshape(mean1, (len(mean1), 1))
    mean2 = np.reshape(np.array(mean2_2), (len(mean2_2), 1))
    x1_2 = np.reshape(np.array(x1_2), (len(x1_2), 1))
    #x1 corresponds to mean1 and x1_2 corresponds to the x's that correspond to mean2
    plt.plot(x1, mean1, 'go')
    plt.plot(x1_2, mean2, 'bo')
    plt.show()

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

if __name__ == "__main__":
    t1()
