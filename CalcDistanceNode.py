import mdp
from numpy import *
from math import sqrt
from LatticeTransformNode import *

def coordsToDistance(x1, y1, z1, x2, y2, z2):
	return sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

class CalcDistanceNode(mdp.Node):
	def is_trainable(self):
		return False

	def _execute(self, coordsArray, lattice_a, lattice_b, lattice_c, alpha, beta, gamma):
		numRows = coordsArray.shape[0]
		numCols = coordsArray.shape[0]
		distanceArray = zeros(shape=(numRows, numCols))
		latticeNode = LatticeTransformNode(lattice_a, lattice_b, lattice_c, alpha, beta, gamma)

		i = 0
		for atomCoord1 in coordsArray:
			j = 0
			centeredCoords = latticeNode(array([atomCoord1]), coordsArray)
			center_coord = centeredCoords[0]
			centeredCoordsArray = centeredCoords[1]
			x1 = center_coord[0][0]
			y1 = center_coord[0][1]
			z1 = center_coord[0][2]
			for atomCoord2 in centeredCoordsArray:
				x2 = atomCoord2[0]
				y2 = atomCoord2[1]
				z2 = atomCoord2[2]
				distanceArray[i][j] = coordsToDistance(x1, y1, z1, x2, y2, z2)
				j = j + 1
			i = i + 1

		return distanceArray
