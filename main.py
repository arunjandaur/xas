import sys
import numpy as np
import mdp
import os

from CalcDistanceNode import CalcDistanceNode
from CalcAngleNode import CalcAngleNode
from PermutationNode import PermutationNode
from parse_intens import parse_intens

SNAPSHOTS_FOLDER_NAME = "snapshots"
OUTPUT_FOLDER = "dist_and_intens"
lattice_a = 32 #8.99341
lattice_b = 32 #8.99341
lattice_c = 32 #8.99341
alpha = 90.0
beta = 90.0
gamma = 90.0

def merge(master, incoming):
	if master == None:
		master = incoming
	else:
		for key1 in master:
			for key2 in master[key1]:
				for key3 in master[key1][key2]:
					master[key1][key2][key3] = np.vstack((master[key1][key2][key3], incoming[key1][key2][key3]))
	return master

def expand(matr):
	matr = np.transpose(matr)
	d1 = matr[0]
	d2 = matr[1]
	a = matr[2]
	cosa = np.cos(a)
	sina = np.sin(a)
	matr = np.transpose(np.vstack((d1, d2, a)))
	return matr

def extractData():
	"""
	General description:
		Loops through all snapshots (various xyz files)
		Loads the coordinates and generates a distance matrix by using the CalcDistanceNode (lattice algebra is taken care of inside CalcDistanceNode. It uses LatticeTransformNode)
	"""
	#Begin initialization
	atomLabels = np.empty((0, 1))
	intens = np.empty((0, 1))

	snapshots = os.listdir(SNAPSHOTS_FOLDER_NAME)
	firstSnap = snapshots[0]
	firstSnapCoords = np.loadtxt(SNAPSHOTS_FOLDER_NAME + '/' + firstSnap, skiprows=2, usecols=(1, 2, 3))

	distArr = np.empty((0, len(firstSnapCoords))) 
	distNode = CalcDistanceNode()

	angleMaster = None
	angleNode = CalcAngleNode()
	#End initialization

	for snap in snapshots:
		snap_path = SNAPSHOTS_FOLDER_NAME + '/' + snap
		currSnapCoords = np.loadtxt(snap_path, skiprows=2, usecols=(1, 2, 3)) #Parse xyz coordinates
		
		#Labels
		currSnapAtomLabels = np.loadtxt(snap_path, dtype=str, skiprows=2, usecols=(0,)) #Parse atoms labels and stack them
		currSnapAtomLabels = np.reshape(currSnapAtomLabels, (len(currSnapAtomLabels), 1))
		atomLabels = np.vstack((atomLabels, currSnapAtomLabels))
		#Labels
		
		#Distances
		currSnapDistArr = distNode(currSnapCoords, lattice_a, lattice_b, lattice_c, alpha, beta, gamma) #Generate distance matrix
		distArr = np.vstack((distArr, currSnapDistArr)) #Accumulate into main distance matrix
		#Distances

		#Angles
		currSnapAngleMaster = angleNode(currSnapCoords, currSnapAtomLabels, lattice_a, lattice_b, lattice_c, alpha, beta, gamma)
		angleMaster = merge(angleMaster, currSnapAngleMaster)
		#Angles
		
		#Intensities
		currSnapIntens = np.empty((0, 1))
		for atomNum in range(len(currSnapAtomLabels)): #Loop through each atom in the snapshot and parse the intens associated with that atom
			atomName = currSnapAtomLabels[atomNum][0]
			atomIntens = parse_intens(atomName, atomNum+1, snap)
			currSnapIntens = np.vstack((currSnapIntens, atomIntens)) #Accumulate intens of all atoms in this snap
		intens = np.vstack((intens, currSnapIntens)) #Accumulate the intens of all snaps
		#Intensities
	
	#Distances
	permuteNode = PermutationNode() #Now split the distances into a nested dictionary where atoms are the keys and the values associated with the inner atoms keys contain the distance matrix between the outer atom key and the inner atom key. The other value associated with the outer atom (along with the inner dictionary described above) is the instensity matrix. The inner dictionary and intensity matrix are bundled in an array
	distanceMaster = permuteNode(distArr, atomLabels, intens)
	#Distances

	return distanceMaster, angleMaster, intens

if __name__ == "__main__":
	dists, angles, intens = extractData()
	
