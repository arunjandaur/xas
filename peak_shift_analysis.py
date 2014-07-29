from __future__ import division
from numpy import random
from math import *
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
import os

def gauss_creator_simple(num_of_gauss):
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

def gauss_creator_complex(num_of_gauss, num_of_variables):
	"""
	Higher order function that returns a gaussian curve function 
	with the number of gaussians specified
	where the mean is a function of the variables given
	(num of variables does not include the constant offset,
	so if it is set to 0 then there are not extra variables and the mean
	is constant)

	The variable inputs will be of the form
	x_position, amplitude1, sigma1, a1, b1...amplitude2, sigma2, a2, b2, etc
	"""
	if num_of_gauss <= 0:
		raise ValueError("gauss_creator needs a nonzero positive num of gaussians")
	if num_of_variables <= 0:
		raise ValueError("gauss_creator_complex needs a nonzero positive num of extra variables, if you want it to be a constant value, use gauss_simple")

	param_num = num_of_variables + 1

	def mean_func(x, *params):
		if len(params) != param_num:
			if len(params) > param_num:
				raise ValueError("too many params for mean function, can only take {}".format(param_num))
			else:
				raise ValueError("too few params for mean function, can only take {}".format(param_num))

		copy = x.copy()
		if x.shape[1] != num_of_variables:
			raise ValueError("input array does not have enough variables") 
		copy_con = np.insert(x, 0,1, axis=1)
		return copy_con.dot(params)

	def make_func(func):
		return lambda E, A, sigma, *args: func(E, *(args[param_num:])) + A * np.exp(-.5 * np.power((E[:,0]-mean_func(E[:,1:],*(args[:param_num])))/sigma, 2))

	func = lambda E, A, sigma, *args : A * np.exp(-.5 * np.power((E[:,0]-mean_func(E[:,1:],*(args[:param_num]))) / sigma, 2))

	for _ in range(num_of_gauss-1):
		func = make_func(func)
	return func

def gauss_creator_minimal(num_gauss, num_vars, amps, sigmas):
	"""
	Returns a gaussian function that takes in only the mean coefficients, since the amplitudes and sigmas have already been estimated using estimate_num_gauss.
	"""
	if num_gauss <= 0:
                raise ValueError("gauss_creator needs a nonzero positive num of gaussians")
        if num_vars <= 0:
                raise ValueError("gauss_creator_complex needs a nonzero positive num of extra variables, if you want it to be a constant value, use gauss_simple")

        param_num = num_vars + 1
        def mean_func(x, *params):
                if len(params) != param_num:
                        if len(params) > param_num:
                                raise ValueError("too many params for mean function, can only take {}".format(param_num))
                        else:
                                raise ValueError("too few params for mean function, can only take {}".format(param_num))

		copy = x.copy()
                if x.shape[1] != num_vars:
                        raise ValueError("input array does not have enough variables")
                copy_con = np.insert(x, 0, 1, axis=1) #Inserts a column of 1's so that we can have [1's] * CONST + [coeffs] * X (vars)
                return copy_con.dot(params)

        def make_func(func, A, sigma):
                return lambda E, *args: func(E, *(args[param_num:])) + A * np.exp(-.5 * np.power((E[:, 0]-mean_func(E[:, 1:], *(args[:param_num]))) / sigma, 2))

	A = amps[0]
	sigma = sigmas[0]
        func = lambda E, *args : A * np.exp(-.5 * np.power((E[:, 0] - mean_func(E[:, 1:], *(args[:param_num]))) / sigma, 2))

        for i in range(num_gauss-1):
                func = make_func(func, amps[i+1], sigmas[i+1])
        return func

def smooth_gaussians(data, sigmas):
	"""
	Convolve data with 1d gaussian kernels of size sigma for all sigma in sigmas
	INPUT: data are sigmas are 1d arrays
	OUTPUT: Outputs convolution of the 0th and 2nd order.
	"""
	#TODO: Indexing error checking
	convolved_0 = np.empty(data.shape)
	convolved_2 = np.empty(data.shape)
        gaussian_filter1d(data, sigma=sigmas[0], output=convolved_0, mode='reflect')
	gaussian_filter1d(data, sigma=sigmas[0], order=2, output=convolved_2, mode='reflect')
	retval = convolved_0
	retval2 = convolved_2
        for sig in sigmas[1:]:
        	gaussian_filter1d(data, sigma=sig, output=convolved_0, mode='reflect')
		gaussian_filter1d(data, sigma=sig, order=2, output=convolved_2, mode='reflect')
        	retval = np.vstack((retval, convolved_0))
		retval2 = np.vstack((retval2, convolved_2))
	return retval, retval2

def get_zero_crossings(input_data, output_data):
	"""
	Finds the points at which positive becomes negative, vice versa, or positive/negative becomes 0. If positive becomes negative or vice versa then the left value is selected. Even though that is not the zero crossing it will be close enough, given that the data was sufficiently sampled.
	OUTPUT: The energies at which the zero crossings approximately occurs. Accuracy is based on sampling rate. The zero crossing will be at most 1 delta x off, where delta x is the sampling width.
	"""
	#TODO: index out of bounds error checking
	left = output_data[:, 0:output_data.shape[1]-3]
	right = output_data[:, 1:output_data.shape[1]-2]
	right2 = output_data[:, 2:output_data.shape[1]-1]

	zero_crossings = []
        for i in range(len(left)):
                row_zeros = []
                for j in range(len(left[0])):
                        left_val = left[i][j]
                        right_val = right[i][j]
                        right2_val = right2[i][j]
                        if right_val == 0 and ((left_val < 0 and right2_val > 0) or (left_val > 0 and right_val < 0)):
				row_zeros.append(input_data[j+1])
                        elif (left_val < 0 and right_val > 0) or (left_val > 0 and right_val < 0):
				row_zeros.append(input_data[j])
                zero_crossings.append(row_zeros)
        return np.array(zero_crossings)

def to_arc_space(zeros, sigmas):
	"""
	Only useful for plotting purposes. This converts rows of zeroes, where the ith row has the crossing locations when convolved with a 1d gaussian kernel with sigma = sigmas[i].
	OUTPUT: Rows which contain 2 values each: The location and sigma at which the crossing appears. Each row is a separate crossing.
	"""
	arc_data = []
	for i in range(len(sigmas)):
		sigma = sigmas[i]
		for zero in zeros[i]:
			arc_data.append([zero, sigma])
	return np.array(arc_data)

def find_closest_crossing(val, crossings):
	"""
	Takes an energy value and finds the closest crossing in crossings to val (distance metric is just horizontal distance).
	INPUT: val is a float.
	OUTPUT: crossings is a 1d array of energy values of crossings at a particular sigma.
	"""
	min_dist, min_crossings = 40, 0
        for cross in crossings:
        	if abs(cross - val) < min_dist:
                	min_dist = abs(cross - val)
			min_crossing = cross
	return min_crossing

def find_pairs(pairs, crossings):
	"""
	Matches pairs with crossings and takes the leftover crossings and makes them into pairs
	INPUT: pairs has rows where each row has two elements, the left and right zero crossings that denote a gaussian. crossings is a 1d array of energy positions of zero crossings taken at a different kernel sigma than ones in pairs.
	OUTPUT: Rows where each row is a pair of crossings that most closely matched the pair in pairs at the corresponding index. Returns False if something went wrong.
	"""
	#TODO: Fix problem when crossings is empty
	if len(pairs) * 2 > len(crossings):
		print "Insufficient crossings!"
		return np.array(pairs)
	new_pairs = []
	copy = list(np.copy(crossings))
	for i in range(len(pairs)):
		pair = pairs[i]
		left, right = pair[0], pair[1]
		min_left = find_closest_crossing(left, copy)
		copy.remove(min_left)
		min_right = find_closest_crossing(right, copy)
		copy.remove(min_right)
		new_pairs.append([min_left, min_right])
	if len(copy) == 2:
		new_left, new_right = copy[0], copy[1]
		inner = False
		for i in range(len(new_pairs)):
			old_pair = new_pairs[i]
			old_left, old_right = old_pair[0], old_pair[1]
			if new_left > old_left and new_left < old_right and new_right > old_left and new_right < old_right:
				new_pairs[i][1] = new_left
				new_pairs.append([new_right, old_right])
				inner = True
				break
		if inner == False:
			new_pairs.append([copy[0], copy[1]])
	elif len(copy) != 0:
		print pairs, crossings
		return False
	return np.array(new_pairs)

def label_arches(zero_crossings):
	"""
	Pairs zero crossings by looking at decreasingly blurry data and labeling new zero crossings that appear as new Gaussians.
	INPUT: Each row is a list of zero crossings in the second derivative that occured at sigmas[i]
	OUTPUT: A 2d array where each row is a pair of crossings that represents a Gaussian. The first crossings to appear are placed towards the left side of the array. The 0th pair showed first, 1st appeared second, etc.
	"""
	i = len(zero_crossings)-1
	prev_pairs = []
	while i >= 0:
		crossings = zero_crossings[i]
		prev_pairs = find_pairs(prev_pairs, crossings)
		if type(prev_pairs) == bool:
			return False
		i -= 1
	return prev_pairs

def estimate_means(arches):
	"""
	INPUT: Each row in arches is a pair of zero crossings of the second derivative. Earlier indexed arches appeared earlier.
	OUTPUT: 1d array of means corresponding to the same index as the input.
	"""
	means = []
	for arch in arches:
		left, right = arch[0], arch[1]
		mean = (left + right) / 2.0
		means.append(mean)
	return np.array(means)

def estimate_sigmas(arches):
	"""
	Same as estimate_means
	"""
	sigmas = []
	for arch in arches:
		left, right = arch[0], arch[1]
		sigma = abs(left - right) / 2.0
		sigmas.append(sigma)
	return np.array(sigmas)

def estimate_amplitudes(input_data, output_data, means, sigmas):
	num_gauss = len(means)
        func = lambda X, avg, sigma : np.exp(-.5 * np.power((X-avg) / sigma, 2))
        coef_table = func(input_data, means[0],sigmas[0])
        amps = []
        if num_gauss > 1:
            for mean,sigma in zip(means[1:],sigmas[1:]):
                    temp = func(input_data, mean,sigma)
                    coef_table = np.column_stack((coef_table,temp))
        if len(coef_table.shape) == 1:
            coef_table.shape += (1,)
        return scp.linalg.lstsq(coef_table,output_data)[0] 

def estimate_num_gauss(arches, tol, input_data, output_data):
	"""
	This algorithm fits a variable number of gaussians and estimate their parameters. This is the algorithm from the research paper. It picks an increasing number of gaussians until the error is below a certain tolerance.
	INPUT: arches have the same format as estimate_means. tol is the error tolerance for when the algorithm should stop increasing the number of gaussians to fit.
	OUTPUT: The number of gaussians and their fit parameters (amp, mean, sigma).
	"""
	n, m, error, params = 1, len(arches), 1000, []
	while error > tol:
		if n > m:
			return m #maybe instead we should remember the n with the least error
		n_dominant = arches[0:n]
		gauss_func = gauss_creator_simple(n)
		means = estimate_means(n_dominant)
		sigmas = estimate_sigmas(n_dominant)
		amps = estimate_amplitudes(input_data, output_data, means, sigmas)
		initialparams = []
		for i in range(n):
			initialparams.append(amps[i])
			initialparams.append(means[i])
			initialparams.append(sigmas[i])
		params, covar = curve_fit(gauss_func, input_data, output_data, p0=initialparams, maxfev=4000)
		error = np.sqrt(1/len(input_data) * np.sum(np.power(output_data - gauss_func(input_data, *params), 2)))
		n += 1
	return n-1, params

def estimate_mean_coeffs(means):
	"""
	Estimates the coefficients for sigma(X) = a + bx1 + cx2 + ...
	INPUT: 1d array of means where means at smaller indices appear first (at higher sigmas)
	OUPUT: List of coefficients in the order a, b, c, etc
	"""
	coeffs = []
	for mean in means:
		coeffs.append(0)
		coeffs.append(mean)
	return coeffs

def graph():
	plt.figure(1)
	plt.subplot(221)
        for i in range(len(convolved_2)):
	        plt.plot(input_data[:, 0][0:1000], convolved_2[i], 'b', label='fit')
        plt.plot(input_data[:, 0][0:1000], output_data[0:1000], 'ro', label='original')
        plt.subplot(222)
        plt.plot(arc_data[:, 0], arc_data[:, 1], 'go', label='arc data')
        plt.show()

if __name__ == "__main_":
        """Lev"""
        input_data = np.array([[], [], []])
	X = np.random.normal(loc=1.16, scale=.16, size=100)
	X2 = np.random.normal(loc=3, scale=.1, size=100)
	for i in range(len(X)):
        	x = X[i]
		x2 = X2[i]
                energies = np.linspace(0, 30, 1000)
                x_s = [x for _ in range(len(energies))]
		x2_s = [x2 for _ in range(len(energies))]
                temp = np.vstack((energies, x_s, x2_s))
                input_data = np.hstack((input_data, temp))
	input_data = np.transpose(input_data)
        num_gauss = 1
	gauss_complex = gauss_creator_simple(num_gauss)
        output_data = gauss_complex(input_data[:,0], 5,15,3, 15,6,7)
        print output_data.shape
	output_1 = output_data[0:1000]
        amps = estimate_amplitudes(input_data[:,0][0:1000],output_1,(15,),(3,))
        print amps

if __name__ == "__main__":
	"""
        input_data = np.array([[], [], []])
	X = np.random.normal(loc=1.16, scale=.16, size=100)
	X2 = np.random.normal(loc=3, scale=.1, size=100)
	for i in range(len(X)):
        	x = X[i]
		x2 = X2[i]
                energies = np.linspace(0, 30, 1000)
                x_s = [x for _ in range(len(energies))]
		x2_s = [x2 for _ in range(len(energies))]
                temp = np.vstack((energies, x_s, x2_s))
                input_data = np.hstack((input_data, temp))
	input_data = np.transpose(input_data)
        num_gauss = 2
	num_vars = 2
	gauss_complex = gauss_creator_complex(num_gauss, num_vars)
	output_data = gauss_complex(input_data, .16, .25, 14.5, .50, 1, .26, .25, 15.5, .50, 1)
	output_1 = output_data[0:1000]
        """
        loaded_values = np.loadtxt("./xas/" + os.listdir("./xas/")[0], usecols=(0,1))
        input_1 = loaded_values[:,0][0:700]
        output_1 = loaded_values[:,1][0:700]
        sigmas = np.arange(1, 1000, 1)
	convolved_0, convolved_2 = smooth_gaussians(output_1, sigmas)
	zero_crossings = get_zero_crossings(input_1, convolved_2)
	arc_data = to_arc_space(zero_crossings, sigmas)
        arches = label_arches(zero_crossings)
	
	num, params = estimate_num_gauss(arches, .001, input_1, output_1)
	
        i, amps, means, sigmas = 0, [], [], []
  	while i < len(params)-2:
  		amp = params[i]
  		mean = params[i+1]
  		sigma = params[i+2]
  		amps.append(amp)
  		means.append(mean)
		sigmas.append(sigma)
  		i += 3
        plt.subplot(211)
        plt.plot(input_1, output_1)
        plt.title("original")
        plt.subplot(212)
        plt.plot(input_1, gauss_creator_simple(num)(params))
        plt.title("fitted")
        plt.show()
        """	
	gauss = gauss_creator_minimal(num, num_vars, amps, sigmas)
	newparams = []
	for mean in means:
		newparams.append(mean)
		for _ in range(num_vars):
			newparams.append(0) #These are the mean coefficient estimates. Until we implement something better, we will use the estimated means from the simple gaussian model as the constants and the coefficients as 0's.
	
  	print params
  	print newparams
  	finalparams, covar = curve_fit(gauss, input_data, output_data, p0=newparams, maxfev=4000)
  	print finalparams
        """

def sum_gaussians_fit(input_data, output_data):
	return 3, np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])

	sigmas = np.arange(.01, 50, .01)
	convolved_0, convolved_2 = smooth_gaussians(output_data, sigmas)
	zero_crossings = get_zero_crossings(input_1, convolved_2)
	arches = label_arches(zero_crossings)

	num, params = estimate_num_gauss(arches, .001, input_data, output_data)

	i, amps, means, sigmas = 0, [], [], []
	while i < len(params)-2:
		amp = params[i]
		mean = params[i+1]
		sigma = params[i+2]
		amps.append(amp)
		means.append(mean)
		sigmas.append(sigma)
		i += 3

	return num, np.array(amps), np.array(means), np.array(sigmas)
