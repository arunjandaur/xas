from sklearn import decomposition
import numpy
from mdp.nodes import FANode
from numpy.linalg import inv

def varimax(Phi, gamma = 1, q = 50, tol = 1e-6):
	from numpy import eye, asarray, dot, sum, diag
	from numpy.linalg import svd
	p,k = Phi.shape
	R = eye(k)
	d=0
	for i in xrange(q):
		d_old = d
		Lambda = dot(Phi, R)
		u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
		R = dot(u,vh)
		d = sum(s)
		if d/d_old < tol: break

	return dot(Phi, R)

def practice(matr):
	numpy.set_printoptions(suppress=True)#, precision=3)
	FA = FANode(tol=0.0001, max_cycles=500, verbose=False, input_dim=None, output_dim=None, dtype=float)
	FA(matr)
	#print "A:\n", FA.A
	#print "A^-1:\n", inv(FA.A)
	V = varimax(FA.A)
	print "V:\n", V
	print "V^-1:\n", inv(V)

"""
matr = [[3.0, 6, 5],
	[7, 3, 3  ],
	[10, 9, 8 ],
	[3, 9, 7  ],
	[10, 6, 5 ]]
matr = numpy.array(matr)
practice(matr)
"""
