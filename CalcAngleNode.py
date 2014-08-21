from __future__ import division
from math import sqrt, acos

import mdp
import numpy as np
from LatticeTransformNode import LatticeTransformNode

class CalcAngleNode(mdp.Node):
    def is_trainable(self):
        return False

    def _dot(self, coord1, coord2):
        return coord1[0] * coord2[0] + coord1[1] * coord2[1] + coord1[2] * coord2[2]

    def _sub(self, coord1, coord2):
        x = coord1[0] - coord2[0]
        y = coord1[1] - coord2[1]
        z = coord1[2] - coord2[2]
        return np.array([x, y, z])

    def _mag(self, coord):
        return sqrt(coord[0] ** 2 + coord[1] ** 2 + coord[2] ** 2)

    def _computeAngle(self, center, a, b):
        #Returns angle in radians
        if (center == a).all() or (center == b).all() or (a == b).all():
            print 'hi'
            return np.inf

        sub_a_center = self._sub(a, center)
        sub_b_center = self._sub(b, center)
        a_dot_b = self._dot(sub_a_center, sub_b_center)
        mag_a_mag_b = self._mag(sub_a_center) * self._mag(sub_b_center)
        return acos(a_dot_b / mag_a_mag_b)

    def _execute(self, coords, *args):
        coords_len = len(coords)
        angles = np.empty((coords_len, coords_len, coords_len))
        
        if len(args) != 0:
            lattice_a, lattice_b, lattice_c, alpha, beta, gamma = args
            latticeNode = LatticeTransformNode(lattice_a, lattice_b, lattice_c, alpha, beta, gamma)
            for i in range(len(coords)):
                center, neighbors = latticeNode(np.array([coords[i]]), np.vstack((coords[0:i], coords[i+1:])))
                coords2 = np.vstack((neighbors[0:i], center, neighbors[i:]))
                for j in range(len(coords2)):
                    for k in range(len(coords2)):
                        angle = self._computeAngle(center[0], coords2[j], coords2[k])
                        angles[i][j][k] = angle
            return angles

        for i in range(len(coords)):
            for j in range(len(coords)):
                for k in range(len(coords)):
                    angles[i][j][k] = self._computeAngle(coords[i], coords[j], coords[k])
        return angles
