from __future__ import division
from math import sqrt, acos

import mdp
import numpy as np
from LatticeTransformNode import LatticeTransformNode

class CalcAngleNode(mdp.Node):
    def is_trainable(self):
        return False

    def _dot(self, coord1, coord2):
        return coord1[0][0] * coord2[0][0] + coord1[0][1] * coord2[0][1] + coord1[0][2] * coord2[0][2]

    def _sub(self, coord1, coord2):
        x = coord1[0][0] - coord2[0][0]
        y = coord1[0][1] - coord2[0][1]
        z = coord1[0][2] - coord2[0][2]
        return np.array([[x, y, z]])

    def _mag(self, coord):
        return sqrt(coord[0][0] ** 2 + coord[0][1] ** 2 + coord[0][2] ** 2)

    def _computeAngle(self, coord1, coord2, coord3):
        #Returns angle in radians
        center = coord1
        a = coord2
        b = coord3
        sub_a_center = self._sub(a, center)
        sub_b_center = self._sub(b, center)
        a_dot_b = self._dot(sub_a_center, sub_b_center)
        mag_a_mag_b = self._mag(sub_a_center) * self._mag(sub_b_center)
        return acos(a_dot_b / mag_a_mag_b)

    def _execute(self, coordsArray, atomLabels, lattice_a, lattice_b, lattice_c, alpha, beta, gamma):
        angle_matr = {}
        latticeNode = LatticeTransformNode(lattice_a, lattice_b, lattice_c, alpha, beta, gamma)

        for i in range(len(coordsArray)):
            coord1 = coordsArray[i]
            atom1 = atomLabels[i][0]
            centeredCoords = latticeNode(np.array([coord1]), np.vstack((coordsArray[0:i], coordsArray[i+1:]))) #Center coordinates around coord1. Returns 2-elem tuple.
            tempAtomLabels = np.vstack((atomLabels[0:i], atomLabels[i+1:]))
            center_coord = centeredCoords[0][0] #The center coordinate. First index returns 2d array. 2nd index gets first (only) row.
            centeredCoordsArray = centeredCoords[1] #Coordinates surrounding center_coord. Index gets tuple's 2nd elem: 2d array of coords

            for j in range(len(centeredCoordsArray)-1):
                coord2 = centeredCoordsArray[j]
                atom2 = tempAtomLabels[j][0]
                for k in range(len(centeredCoordsArray[j+1:])):
                    coord3 = centeredCoordsArray[k+j+1]
                    atom3 = tempAtomLabels[k+j+1][0]
                    angle = self._computeAngle(np.array([center_coord]), np.array([coord2]), np.array([coord3]))
                    #print atom1, atom2, atom3
                    if atom1 not in angle_matr:
                        angle_matr[atom1] = {atom2: {atom3: np.array([[angle]])}}
                    elif atom2 not in angle_matr[atom1]:
                        angle_matr[atom1][atom2] = {atom3: np.array([[angle]])}
                    elif atom3 not in angle_matr[atom1][atom2]:
                        angle_matr[atom1][atom2][atom3] = np.array([[angle]])
                    else:
                        angle_matr[atom1][atom2][atom3] = np.vstack((angle_matr[atom1][atom2][atom3], np.array([[angle]])))
                    k += 1
                j += 1
            i += 1
        return angle_matr
