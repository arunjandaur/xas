import numpy as np
from numpy import arange, array, ones, linalg
from pylab import plot, show
from mdp.nodes import PCANode

def lin_reg_test():
	xi = arange(0,9)
	A = array([ xi, ones(9)])
	#print A
	#print A.T
	# linearly generated sequence
	y = [19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24]
	w = linalg.lstsq(A.T,y)[0] # obtaining the parameters
	
	# plotting the line
	line = w[0]*xi+w[1] # regression line
	print w[1]
	plot(xi,line,'r-',xi,y,'o')
	show()

def plot_line(coeffs):
	line 
	pass
	

def lin_reg(coords, intensities):
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
	ones_column = np.array([ones(len(coords))]).T
	coords = np.hstack((coords, ones_column)) #matrix variable instantiations to be right multiplied by coeff matrix to obtain intensities. Goal is to find coeff matrix.
	print intensities
	coeffs = linalg.lstsq(coords, intensities)[0] #1st elem of tuple. In each column (corresponding to an energy), each row represents a coefficient to fit the coordinates to the intensity at a certain energy. The last row is the constant b.
	print 'hi', coeffs
	
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
