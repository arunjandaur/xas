import mdp
from numpy import *
from math import sqrt
from LatticeTransformNode import *

def coordsToDistance(x1, y1, z1, x2, y2, z2):
    """
    Input: 6 floats that represent the xyz coordinates of atom 1 and atom 2
    Output: 1 float that represents the euclidean distance between atom 1 and atom 2
    """
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
            i = 0
            for atomCoord1 in coords:
                j = 0
                centeredCoords = latticeNode(array([atomCoord1]), coords) #Center the coordinates around atomCoord1
                center_coord = centeredCoords[0] #The centered coordinate
                centeredCoordsArray = centeredCoords[1] #The coordinates surrounding center_coord
                x1 = center_coord[0][0] #Get x, y, and z
                y1 = center_coord[0][1]
                z1 = center_coord[0][2]
                for atomCoord2 in centeredCoordsArray: #for each element j in J, compute dist(i, j)
                    x2 = atomCoord2[0]#Get x, y, and z
                    y2 = atomCoord2[1]
                    z2 = atomCoord2[2]
                    dists[i][j] = coordsToDistance(x1, y1, z1, x2, y2, z2)
                    j = j + 1
                i = i + 1

            return dists

        for i in range(len(coords)):
            x1, y1, z1 = coords[i]
            for j in range(len(coords)):
                x2, y2, z2 = coords[j]
                dists[i][j] = coordsToDistance(x1, y1, z1, x2, y2, z2)
        return dists
