import sys
import numpy as np
import mdp
import os

from mdp.nodes import PCANode

from CalcDistanceNode import CalcDistanceNode
from PermutationNode import PermutationNode
from parse_intensities import parse_intensities

import cluster
import analysis

SNAPSHOTS_FOLDER_NAME = "snapshots"
OUTPUT_FOLDER = "dist_and_intens"

lattice_a = 32 #8.99341
lattice_b = 32 #8.99341
lattice_c = 32 #8.99341
alpha = 90.0
beta = 90.0
gamma = 90.0

if __name__ == '__main__':
	"""
	General description:
		Loops through all snapshots (various xyz files)
		Loads the coordinates and generates a distance matrix by using the CalcDistanceNode (lattice algebra is taken care of inside CalcDistanceNode. It uses LatticeTransformNode)
	"""
	#Begin initialization
	atomLabels = np.empty((0, 1))
	intensities = np.empty((0, 1))

	snapshots = os.listdir(SNAPSHOTS_FOLDER_NAME)
	firstSnap = snapshots[0]
	firstSnapCoords = np.loadtxt(SNAPSHOTS_FOLDER_NAME + '/' + firstSnap, skiprows=2, usecols=(1, 2, 3))
	distanceArray = np.empty((0, len(firstSnapCoords))) 
	calcDistanceNodeInst = CalcDistanceNode()
	#End initialization

	for snap in snapshots:
		snap_path = SNAPSHOTS_FOLDER_NAME + '/' + snap #Construct filepath
		currentSnapCoords = np.loadtxt(snap_path, skiprows=2, usecols=(1, 2, 3)) #Parse xyz coordinates
		currentSnapDistanceArray = calcDistanceNodeInst(currentSnapCoords, lattice_a, lattice_b, lattice_c, alpha, beta, gamma) #Generate distance matrix
		distanceArray = np.vstack((distanceArray, currentSnapDistanceArray)) #Accumulate into main distance matrix
		
		currentSnapAtomLabels = np.loadtxt(snap_path, dtype=str, skiprows=2, usecols=(0,)) #Parse atoms labels and stack them
		currentSnapAtomLabels = np.reshape(currentSnapAtomLabels, (len(currentSnapAtomLabels), 1))
		atomLabels = np.vstack((atomLabels, currentSnapAtomLabels))

		currentSnapIntensities = np.empty((0, 1))
		for atomNum in range(len(currentSnapAtomLabels)): #Loop through each atom in the snapshot and parse the intensities associated with that atom. It uses the functionality from parse_intensities.py
			atomName = currentSnapAtomLabels[atomNum][0]
			atomIntensities = parse_intensities(atomName, atomNum+1, snap)
			currentSnapIntensities = np.vstack((currentSnapIntensities, atomIntensities)) #Accumulate intensities of all atoms in this snap
			
		intensities = np.vstack((intensities, currentSnapIntensities)) #Accumulate the intensities of all snaps

	permuteNode = PermutationNode() #Now split the distances into a nested dictionary where atoms are the keys and the values associated with the inner atoms keys contain the distance matrix between the outer atom key and the inner atom key. The other value associated with the outer atom (along with the inner dictionary described above) is the instensity matrix. The inner dictionary and intensity matrix are bundled in an array
	permutedDistanceArray = permuteNode(distanceArray, atomLabels, intensities)

	#Now write each distance matrix corresponding to a different atom-atom pair to its own file for later use. TODO: pickle the matrix
	master = permutedDistanceArray
	if not os.path.exists(OUTPUT_FOLDER):
		os.makedirs(OUTPUT_FOLDER)
	for atom in master:
		for atom2 in master[atom][0]:
			distArray = master[atom][0][atom2]
			intenArray = master[atom][1]
			distFileName = OUTPUT_FOLDER + '/' + atom + "-" + atom2 + '.txt'
			intenFileName = OUTPUT_FOLDER + '/' + atom + "-" + atom2 + "_xas" + '.txt'

			distFile = open(distFileName, 'w')
			distFile.write(str(distArray))
			intenFile = open(intenFileName, 'w')
			intenFile.write(str(intenArray))
	
	S_S_data = master['C'][0]['O']
	S_S_intensities = master['C'][1]
	
	print "DISTANCE MATRIX"
	print S_S_data
	print "INTENSITIES"
	print S_S_intensities
	
	#cluster.oneD_cluster(S_S_data)	
	
	pca = PCANode(reduce=True)
	result = pca.execute(S_S_data)
	print pca.d
	print pca.v
	
	#analysis.lin_reg(S_S_data, S_S_intensities)
