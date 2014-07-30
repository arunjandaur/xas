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
                raise ValueError("gauss_creator_minimal needs a nonzero positive num of extra variables, if you want it to be a constant value, use gauss_simple")

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
                return lambda X, *args: func(X, *(args[param_num:])) + A * np.exp(-.5 * np.power((X[:, 0]-mean_func(X[:, 1:], *(args[:param_num]))) / sigma, 2))

	A = amps[0]
	sigma = sigmas[0]
        func = lambda X, *args : A * np.exp(-.5 * np.power((X[:, 0] - mean_func(X[:, 1:], *(args[:param_num]))) / sigma, 2))

        for i in range(num_gauss-1):
                func = make_func(func, amps[i+1], sigmas[i+1])
        return func

def smooth_gaussians(data,sigmas, order=0):
	"""
	Convolve data with 1d gaussian kernels of size sigma for all sigma in sigmas
	INPUT: data are sigmas are 1d arrays
	OUTPUT: Outputs convolution of the 0th and 2nd order.
	"""
	#TODO: Indexing error checking
	convolved = np.empty(data.shape)
	gaussian_filter1d(data, sigma=sigmas[0], order=order, output=convolved, mode='reflect')
	retval = convolved.copy()
        for sig in sigmas[1:]:
		gaussian_filter1d(data, sigma=sig, order=order, output=convolved, mode='reflect')
        	retval = np.vstack((retval, convolved))
        return retval

def get_zero_crossings(input_data, output_data):
	"""
	Finds the points at which positive becomes negative, vice versa, or positive/negative becomes 0. If positive becomes negative or vice versa then the left value is selected. Even though that is not the zero crossing it will be close enough, given that the data was sufficiently sampled.
	OUTPUT: The energies at which the zero crossings approximately occurs. Accuracy is based on sampling rate. The zero crossing will be at most 1 delta x off, where delta x is the sampling width.
	"""
        zero_crossings = []
        if len(output_data.shape) == 1:
            left = output_data[0:output_data.shape[1]-2]
            right = output_data[1:output_data.shape[1]-1]
            right2 = output_data[2:output_data.shape[1]]

            for j in range(len(left[0])):
                    left_val = left[j]
                    right_val = right[j]
                    right2_val = right2[j]
                    if right_val == 0 and ((left_val < 0 and right2_val > 0) or (left_val > 0 and right_val < 0)):
                            zero_crossings.append(input_data[j+1])
                    elif (left_val < 0 and right_val > 0) or (left_val > 0 and right_val < 0):
                            zero_crossings.append(input_data[j])
        else:
            left = output_data[:, 0:output_data.shape[1]-2]
            right = output_data[:, 1:output_data.shape[1]-1]
            right2 = output_data[:, 2:output_data.shape[1]]

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

def find_closest_crossing(val, zeros):
	"""
	Takes an energy value and finds the closest zero_crossing in zeros to val (distance metric is just horizontal distance).
	INPUT: val is a float.
	OUTPUT: crossings is a 1d array of energy values of crossings at a particular sigma.
	"""
	min_dist, min_zero = float("inf"), 0
        for zero in zeros:
        	if abs(zero - val) < min_dist:
                	min_dist = abs(zero - val)
			min_zero = zero
	return min_zero

def find_pairs(pairs, zeros, singles, input_data, convolved_1):
	"""
	Matches pairs with zeros and takes the leftover zeros and makes them into pairs
	INPUT: pairs has rows where each row has two elements, the left and right zero zeros that denote a gaussian. zeros is a 1d array of energy positions of zero zeros taken at a different gaussian sigma than ones in pairs.
	OUTPUT: Rows where each row is a pair of zeros that most closely matched the pair in pairs at the corresponding index. Returns False if something went wrong.
	"""
	#TODO: Fix problem when zeros is empty

	#CASE 1: NOT ENOUGH CROSSINGS
	if len(pairs) * 2 > len(zeros):
		print "Insufficient zeros!"
		return np.array(pairs), np.array(singles)
	
	#CASE 2: MATCH NEW CROSSINGS WITH PREVIOUS PAIRS
	new_pairs = []
	copy = list(np.copy(zeros))
	for pair in pairs:
		left, right = pair[0], pair[1]
		min_left = find_closest_crossing(left, copy)
		copy.remove(min_left)
		min_right = find_closest_crossing(right, copy)
		copy.remove(min_right)
		new_pairs.append([min_left, min_right])

	#CASE 3: MATCH NEW SINGLES WITH PREVIOUS SINGLES
	new_singles = []
        singles_left_copy = list(copy)
        for single in singles:
                #single_match = find_closest_crossing(single, copy)
                #copy.remove(single_match)
	        if len(singles_left_copy) == 0:
                    break
                single_match = find_closest_crossing(single, singles_left_copy)
                #print "singles_left_copy is {}".format(singles_left_copy)
                #print "single_match is {}".format(single_match)
                #print "singles is {}".format(singles)
                singles_left_copy.remove(single_match)
		new_singles.append(single_match)

	#CASE 4: LEFTOVER CROSSINGS
	leftovers = len(copy)
        #cross_index = np.scalar(np.argwhere(cross==input_data))
        while len(copy) > 0:
            zero = copy[0]
            zero_index = np.asscalar(np.argwhere(zero==input_data))
            #print "zero_index is {}".format(zero_index)
            first_deriv = convolved_1[zero_index]
            #print "first_deriv is {}".format(first_deriv)
            if first_deriv > 0:
                #ugly. Im modifying list while looping through it. You disgust me, Lev.
                for second_zero in copy[1:]:
                    zero_index_2 = np.asscalar(np.argwhere(second_zero==input_data))
                    #print "zero_index_2 is {}".format(zero_index_2)
                    first_deriv_2 = convolved_1[zero_index_2]
                    #print "first_deriv_2 is {}".format(first_deriv_2)
                    if first_deriv_2 < 0:
                        new_pairs.append([zero,second_zero])
                        copy.remove(second_zero)
                        if zero in new_singles:
                            new_singles.remove(zero)
                        if second_zero in new_singles:
                            new_singles.remove(second_zero)
                        break
                else:
                    if not zero in new_singles:
                        new_singles.append(zero)
                del copy[0]
            elif first_deriv < 0:
                del copy[0]
                if not zero in new_singles:
                    new_singles.append(zero)
            else:
                raise ValueError("why is first derivate zero where there is a second derivative zero crossing. If you ever come accross this error, assume that your calculation of first and second derivatives have been erroneous or we have been working under the wrong assumptions where the first and second derivative zeros are")
	#print new_pairs
        #print new_singles
        return np.array(new_pairs), np.array(new_singles)

def label_arches(zero_crossings, input_data, convolved_1):
	"""
	Pairs zero crossings by looking at decreasingly blurry data and labeling new zero crossings that appear as new Gaussians.
	INPUT: Each row is a list of zero crossings in the second derivative that occured at sigmas[i]
	OUTPUT: A 2d array where each row is a pair of crossings that represents a Gaussian. The first crossings to appear are placed towards the left side of the array. The 0th pair showed first, 1st appeared second, etc.
	"""
        prev_pairs = np.array([])
	singles = np.array([])
	for i in xrange(len(zero_crossings)-1,-1,-1):
	    crossings = zero_crossings[i]
            convolved_1_i = convolved_1[i]
            prev_pairs, singles = find_pairs(prev_pairs, crossings, singles, input_data, convolved_1_i)
        singles = list(singles)
        while len(prev_pairs) < 10 and len(singles) > 1:
            zero = singles[0]
            for single in singles[1:]:
                if single > zero:
                    singles.remove(zero)
                    singles.remove(single)
                    prev_pairs = np.append(prev_pairs,[[zero,single]],0)
                    break
            else:
                singles = list(np.roll(singles,-1))
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
        m, error, params, holder = len(arches), float("inf"), [], []
        for n in xrange(1,m+1):
                print "num of gauss used is {}".format(n)
                #if error < tol:
                    #return i+1,holder[i][1], error
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
                try:
                    params, covar = curve_fit(gauss_func, input_data, output_data, p0=initialparams, maxfev=100000)
		    error = np.sqrt(1/len(input_data) * np.sum(np.power(output_data - gauss_func(input_data, *params), 2)))
                except RuntimeError:
                    params = initialparams
                    error = float("inf")
                
                holder.append([error,params,initialparams])
        
        if True: 
            for i in range(0,len(holder)):
                error = holder[i][0]
                print "error for size {0} is {1}".format(i+1,error)
                plt.subplot(221)
                plt.plot(input_data,output_data)
                plt.title("original")
                plt.subplot(223)
                plt.plot(input_data,gauss_creator_simple(i+1)(input_data,*holder[i][1]))
                plt.title("fitted with new params")
                plt.subplot(224)
                plt.plot(input_data,gauss_creator_simple(i+1)(input_data,*holder[i][2]))
                plt.title("fitted with initial params")
                plt.show()


        while True:
            print "tol is {}".format(tol)
            for i in range(0,len(holder)):
                error = holder[i][0]
                print "error for size {0} is {1}".format(i+1,error)
                if error < tol:
                    return i+1,holder[i][1], error
                if False:
                    plt.subplot(221)
                    plt.plot(input_data,output_data)
                    plt.title("original")
                    plt.subplot(223)
                    plt.plot(input_data,gauss_creator_simple(i+1)(input_data,*holder[i][1]))
                    plt.title("fitted with new params")
                    plt.subplot(224)
                    plt.plot(input_data,gauss_creator_simple(i+1)(input_data,*holder[i][2]))
                    plt.title("fitted with initial params")
                    plt.show()

            tol = tol * 2
            	
        return n, params

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

def remove_odds(crossings):
	"""
	Sometimes zero crossings get pushed out of range and there is initially one 1 zero crossing. This takes care of that by creating the pair when the second appears. Not tested.
	"""
	if len(crossings[-1]) > 3 and len(crossings[-1]) % 2 != 0:
		return False
	i = len(crossings)-1
	cutoff = len(crossings)
	while i >= 0:
		num = len(crossings[i])
		if num > 2 and num % 2 != 0:
			return False
		if num == 2:
			cutoff = i + 1
			break
		i -= 1
	return crossings[:cutoff]

def graph_scale(x,blurred,first,second,sigmas):
        plt.figure()
        for i in range(len(sigmas)):
            if i % 8 == 0:
                plt.legend(bbox_to_anchor = (1,0),loc="lower left")
                plt.show()
            plt.subplot(2,4,(i%8)+1)
            plt.plot(x,blurred[i],"r", label = "blurred")
            plt.plot(x,first[i],"b", label = "first")
            plt.plot( x, second[i], "g", label = "second")
            plt.title("sigma = {}".format(sigmas[i]))
        plt.show()

def graph(input_data, output_data, convolved_2, arc_data):
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
        num_gauss = 2
	gauss_complex = gauss_creator_simple(num_gauss)
        output_data = gauss_complex(input_data[:,0], 5,15,3, 300,6,5)
        input_1  = input_data[:,0][0:1000]
        output_1 = output_data[0:1000]
        sigmas = np.arange(1, 100, 1)
        convolved = smooth_gaussians(output_1, sigmas, order=2)
	zero_crossings = get_zero_crossings(input_1, convolved)
	print [len(cross) for cross in zero_crossings]
	#zero_crossings = remove_odds(zero_crossings)
	arc_data = to_arc_space(zero_crossings, sigmas)
	#graph()
        arches = label_arches(zero_crossings)
        print params
	#print "Arches\n" + str(arches)
	plt.plot(arc_data[:,0],arc_data[:,1],"o")
        plt.show()
        """
        loaded_values = np.loadtxt("./xas/" + os.listdir("./xas/")[0], usecols=(0,1))
        input_1 = loaded_values[:,0][0:700]
        output_1 = loaded_values[:,1][0:700]
        sigmas = np.arange(1, 1000, 1)
        convolved_0, convolved_2 = smooth_gaussians(output_1, sigmas)
        zero_crossings_0 = get_zero_crossings(input_1, np.gradient(np.gradient(convolved_0)[1])[1])
	zero_crossings_2 = get_zero_crossings(input_1, convolved_2)
        print [len(cross) for cross in zero_crossings_0]
        #zero_crossings = remove_odds(zero_crossings)
	arc_data_0 = to_arc_space(zero_crossings_0, sigmas)
        arc_data_2 = to_arc_space(zero_crossings_2, sigmas)
        plt.figure(1)
        plt.subplot(221)
        plt.title("gradient")
        plt.plot(arc_data_0[:,0],arc_data_0[:,1],"o")
        plt.subplot(222)
        plt.title("derivative")
        plt.plot(arc_data_2[:,0],arc_data_2[:,1],"o")
        plt.subplot(223)
        plt.title("original data")
        plt.plot(input_1,output_1)
        plt.show()
        """
if __name__ == "__main_":
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
	graph(input_1, output_1, convolved_2, arc_data)
	#graph(input_1, output_1, convolved_0, arc_data)
        arches = label_arches(zero_crossings, input_1, convolved_1)
	print "Arches\n" + str(arches)

	num, params, error = estimate_num_gauss(arches, .0001, input_1, output_1)
	
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

def example_2():
        input_data = np.linspace(0,30,1000)
        num_gauss = 4
	gauss_simple = gauss_creator_simple(num_gauss)
	output_data = gauss_simple(input_data, .16, 14.5, 0.45, .26, 15.5, .25, .16, 23, 0.20, .26, 25.5, .9)
        sigmas = np.arange(1, 200, 1)
	convolved_1 = smooth_gaussians(output_data, sigmas, order=1)
        convolved_2 = smooth_gaussians(output_data, sigmas, order=2)
	zero_crossings = get_zero_crossings(input_data, convolved_2)
	#print [len(cross) for cross in zero_crossings]
	#zero_crossings = remove_odds(zero_crossings)
	arc_data = to_arc_space(zero_crossings, sigmas)
	#graph()

        arches = label_arches(zero_crossings, input_data, convolved_1)
	#print "Arches\n" + str(arches)
	
	num, params, error = estimate_num_gauss(arches, .0001, input_data, output_data)
        print num
        print params
        i, amps, means, sigmas = 0, [], [], []
  	while i < len(params)-2:
  		amp = params[i]
  		mean = params[i+1]
  		sigma = params[i+2]
  		amps.append(amp)
  		means.append(mean)
		sigmas.append(sigma)
  		i += 3
        plt.subplot(221)
        plt.plot(input_data,output_data)
        plt.title("original")
        plt.subplot(222)
        plt.plot(input_data,gauss_creator_simple(num)(input_data,*params))
        plt.title("fitted")
        plt.subplot(223)
        plt.plot(arc_data[:,0],arc_data[:,1],"o")
        plt.title("scale space zeros")
        plt.subplot(224)
        plt.plot(input_data,output_data - gauss_creator_simple(num)(input_data,*params))
        plt.title("residuals")
        plt.show()

def example_1():
        input_data = np.linspace(0,30,1000)
        num_gauss = 2
	gauss_simple = gauss_creator_simple(num_gauss)
	output_data = gauss_simple(input_data, .16, 14.5, 0.25, .26, 15.5, .25)
        sigmas = np.arange(1, 30, 1)
	convolved_0 = smooth_gaussians(output_data, sigmas, order=0)
        convolved_1 = smooth_gaussians(output_data, sigmas, order=1)
        convolved_2 = smooth_gaussians(output_data, sigmas, order=2)
	zero_crossings = get_zero_crossings(input_data, convolved_2)
	#print [len(cross) for cross in zero_crossings]
	#zero_crossings = remove_odds(zero_crossings)
	arc_data = to_arc_space(zero_crossings, sigmas)
	#graph()
        arches = label_arches(zero_crossings, input_data, convolved_1)
	#print "Arches\n" + str(arches)
        #graph_scale(input_data,convolved_0,convolved_1,convolved_2,sigmas)
        plt.plot(arc_data[:,0],arc_data[:,1],"o")
        plt.show()
        num, params, error = estimate_num_gauss(arches, .0001, input_data, output_data)
        
        i, amps, means, sigmas = 0, [], [], []
  	while i < len(params)-2:
  		amp = params[i]
  		mean = params[i+1]
  		sigma = params[i+2]
  		amps.append(amp)
  		means.append(mean)
		sigmas.append(sigma)
  		i += 3
        plt.subplot(221)
        plt.plot(input_data,output_data)
        plt.title("original")
        plt.subplot(222)
        plt.plot(input_data,gauss_creator_simple(num)(input_data,*params))
        plt.title("fitted")
        plt.subplot(223)
        plt.plot(arc_data[:,0],arc_data[:,1],"o")
        plt.title("scale space zeros")
        plt.subplot(224)
        plt.plot(input_data,output_data - gauss_creator_simple(num)(input_data,*params))
        plt.title("residuals")
        plt.show()

def example_3():
        input_data = np.linspace(0,30,1000)
        num_gauss = 6
	gauss_simple = gauss_creator_simple(num_gauss)
	output_data = gauss_simple(input_data, .16, 14.5, 0.25, .26, 15.5, .25, .16, 16.9, 0.25, .26, 13.6, .25,.30, 10, 0.5, .26, 19, 1)
        sigmas = np.arange(1, 200, 1)
	convolved_1 = smooth_gaussians(output_data, sigmas, order=1)
        convolved_2 = smooth_gaussians(output_data, sigmas, order=2)
	zero_crossings = get_zero_crossings(input_data, convolved_2)
	#print [len(cross) for cross in zero_crossings]
	#zero_crossings = remove_odds(zero_crossings)
	arc_data = to_arc_space(zero_crossings, sigmas)
	#graph()

        arches = label_arches(zero_crossings, input_data, convolved_1)
	#print "Arches\n" + str(arches)
	
	num, params, error = estimate_num_gauss(arches, .0001, input_data, output_data)
        print num
        print params
        i, amps, means, sigmas = 0, [], [], []
  	while i < len(params)-2:
  		amp = params[i]
  		mean = params[i+1]
  		sigma = params[i+2]
  		amps.append(amp)
  		means.append(mean)
		sigmas.append(sigma)
  		i += 3
        plt.subplot(221)
        plt.plot(input_data,output_data)
        plt.title("original")
        plt.subplot(222)
        plt.plot(input_data,gauss_creator_simple(num)(input_data,*params))
        plt.title("fitted")
        plt.subplot(223)
        plt.plot(arc_data[:,0],arc_data[:,1],"o")
        plt.title("scale space zeros")
        plt.subplot(224)
        plt.plot(input_data,output_data - gauss_creator_simple(num)(input_data,*params))
        plt.title("residuals")
        plt.show()

def example_4():
        loaded_values = np.loadtxt("./xas/" + os.listdir("./xas/")[23], usecols=(0,1))
        input_1 = loaded_values[:,0][0:700]
        output_1 = loaded_values[:,1][0:700]
        
        sigmas = np.arange(1, 1000, 1)
	convolved_1 = smooth_gaussians(output_1, sigmas, order=1)
        convolved_2 = smooth_gaussians(output_1, sigmas, order=2)
	zero_crossings = get_zero_crossings(input_1, convolved_2)
	#print [len(cross) for cross in zero_crossings]
	#zero_crossings = remove_odds(zero_crossings)
	arc_data = to_arc_space(zero_crossings, sigmas)
	#graph()

        arches = label_arches(zero_crossings, input_1, convolved_1)
	#print "Arches\n" + str(arches)
	
	num, params, error = estimate_num_gauss(arches, .0001, input_1, output_1)
        
        print num
        print params
        i, amps, means, sigmas = 0, [], [], []
  	while i < len(params)-2:
  		amp = params[i]
  		mean = params[i+1]
  		sigma = params[i+2]
  		amps.append(amp)
  		means.append(mean)
		sigmas.append(sigma)
  		i += 3
        plt.subplot(221)
        plt.plot(input_1,output_1)
        plt.title("original")
        plt.subplot(222)
        plt.plot(input_1,gauss_creator_simple(num)(input_1,*params))
        plt.title("fitted")
        plt.subplot(223)
        plt.plot(arc_data[:,0],arc_data[:,1],"o")
        plt.title("scale space zeros")
        plt.subplot(224)
        plt.plot(input_1,output_1 - gauss_creator_simple(num)(input_1,*params))
        plt.title("residuals")
        plt.show()

if __name__ == "__main__":
        example_4()

def sum_gaussians_fit(input_data, output_data):
	#return 3, np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])
	sigmas = np.arange(.01, 50, .01)
	convolved_0, convolved_2 = smooth_gaussians(output_data, sigmas)
	zero_crossings = get_zero_crossings(input_1, convolved_2)
	arches = label_arches(zero_crossings)

	num, params, error = estimate_num_gauss(arches, .0001, input_data, output_data)

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
