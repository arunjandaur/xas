import mdp
from numpy import *
from math import sqrt
from LatticeTransformNode import *

def _coordsToDistance(coord1, coord2):
    """
    Input: 6 floats that represent the xyz coordinates of atom 1 and atom 2
    Output: 1 float that represents the euclidean distance between atom 1 and atom 2
    """
    x1, y1, z1 = coord1
    x2, y2, z2 = coord2
    return sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

class CalcDistanceNode(mdp.Node):
    def is_trainable(self):
        return False

    def _execute(self, coords, *args):
        """
        Input:
            coords: A 2D array where each row is an array of 3 elements: x, y, and z
            lattice_a,b,c: lattice vectors
            alpha, beta, gamma: lattice angles
        Output:
            A distance array where each (i, j)th element is the distance between atom i and atom j. Note, these coordinates belong to periodic molcules, so we must obey lattice algebra. When computing the distance between any atom i to all other atoms {J}, we must apply the periodicity node (lattice transformation) so that all coordinates are shifted such that atom i is in the middle of the lattice space. Then we can compute all the distances between i and J. Repeat for all i.
        """
        numRows = coords.shape[0]
        numCols = coords.shape[0]
        dists = zeros(shape=(numRows, numCols))

        if len(args) != 0:
            lattice_a, lattice_b, lattice_c, alpha, beta, gamma = args
            latticeNode = LatticeTransformNode(lattice_a, lattice_b, lattice_c, alpha, beta, gamma)

            for i in range(len(coords)):
                center, neighbors = latticeNode(np.array([coords[i]]), np.vstack((coords[0:i], coords[i+1])))
                coords2 = np.vstack((neighbors[0:i], center, neighbors[i:]))
                for j in range(len(coords2)):
                    dists[i][j] = _coordsToDistance(center[0], coords[j])
            return dists

        for i in range(len(coords)):
            for j in range(len(coords)):
                dists[i][j] = _coordsToDistance(coords[i], coords[j])
        return dists
