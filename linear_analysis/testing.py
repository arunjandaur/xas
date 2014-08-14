import numpy as np

def error_calc(l_1, l_2):
	""" Does error calculation on two input lists
	Output: list of average abs error in each axis"""
	error = np.abs(l_1 - l_2)
	ave_error = np.average(error, axis=0)
	return ave_error
