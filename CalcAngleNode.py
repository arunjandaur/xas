import mdp
import numpy as np

from __future__ import division
from math import sqrt, acos

class CalcAngleNode(mdp.Node):
	def is_trainable(self):
		return False

	def dot(self, coord1, coord2):
	        return coord1[0][0] * coord2[0][0] + coord1[0][1] * coord2[0][1] + coord1[0][2] * coord2[0][2]

	def _sub(self, coord1, coord2):
	        x = coord1[0][0] - coord2[0][0]
	        y = coord1[0][1] - coord2[0][1]
	        z = coord1[0][2] - coord2[0][2]
	        return np.array([[x, y, z]])

	def _mag(self, coord1):
		return sqrt(coord[0][0] ** 2 + coord[0][1] ** 2 + coord[0][2] ** 2)

	def _computeAngle(self, coord1, coord2, coord3):
	        #Returns angle in radians
		center = coord1
		a = coord2
		b = coord3
	        return acos(self._dot(self._sub(a, center), self._sub(b, center)) / (self._mag(self._sub(a, center)) * self._mag(self._sub(b, center))))

	def _execute(self, coordsArray, atomLabels):
		angle_matr = {}
		for i in range(len(coordsArray)):
			coord1 = coordsArray[i]
			atom1 = atomLabels[i]
			for j in range(len(coordsArray)):
				coord2 = coordsArray[j]
				atom2 = atomLabels[j]
				for k in range(len(coordsArray)):
					coord3 = coordsArray[k]
					atom3 = atomLabels[k]
					angle = computeAngle(coord1, coord2, coord3)
					
					if atom1 not in angle_matr:
						angle_matr[atom1] = {atom2: {atom3: np.array([[angle]])}}
					else if atom2 not in angle_matr[atom1]:
						angle_matr[atom1][atom2] = {atom3: np.array([[angle]])}
					else if atom3 not in angle_matr[atom1][atom2]:
						angle_matr[atom1][atom2][atom3] = np.array([[angle]])
					else:
						angle_matr[atom1][atom2][atom3].vstack((angle_matr[atom1][atom2][atom3], np.array([[angle]])))
					k += 1
				j += 1
			i += 1
		return angle_matr
