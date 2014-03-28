import numpy as np
from numpy import arange,array,ones,linalg
from pylab import plot,show
from mdp.nodes import PCANode

def lin_reg():
	xi = arange(0,9)
	A = array([ xi, ones(9)])
	print A
	print A.T
	# linearly generated sequence
	y = [19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24]
	w = linalg.lstsq(A.T,y)[0] # obtaining the parameters
	
	# plotting the line
	line = w[0]*xi+w[1] # regression line
	print w[1]
	plot(xi,line,'r-',xi,y,'o')
	show()

if __name__ == '__main__':
	#lin_reg()

	#Begin PCA
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
