import sys
import numpy as np
import mdp
import re
import os

from CalcDistanceNode import CalcDistanceNode
from PermutationNode import PermutationNode

XAS_FOLDER_NAME = "xas"
SNAPSHOTS_FOLDER_NAME = "snapshots"
OUTPUT_FOLDER = "dist_and_intens"
NUM_ENERGIES = 1000

lattice_a = 5
lattice_b = 5
lattice_c = 5
alpha = 90.0
beta = 90.0
gamma = 90.0

def parse_intensities(atomName, atomNum, xyz_name):
	"""
	Takes the name of an xyz snapshot, an atom name (label), and which line the atom is found on, and then constructs the file name of the corresponding xas file. It then takes the 1000 energy intensity key value pairs in that xas file and converts them into a dictionary, which exists inside of a 2D numpy array. 
	Invariant: XYZ file must end in .xyz. Otherwise, xyzWithSnapNum = xyz_name[0:len(xyz_name)-4] will fail. It depends on the last four characters to be .xyz
	"""
	"""
	if atomNum < 10:
		atomNum = '00' + str(atomNum)
	elif atomNum < 100:
		atomNum = '0' + str(atomNum)
	else:
		atomNum = str(atomNum)
	"""
	xyzWithSnapNum = xyz_name[0:len(xyz_name)-4]
	xyzWithoutSnapNum = re.sub(r'(.*)_[0-9]+[.]xyz', r'\1', xyz_name)
	xas_filename = XAS_FOLDER_NAME + '/' + xyzWithSnapNum + '-' + xyzWithoutSnapNum + '.' + atomName + str(atomNum) + '-XCH.xas.5.xas'
	
	intensities = np.array([[{}]])
	if os.path.isfile(xas_filename) == True:
		intensities[0][0] = 2
		#raw_xas = np.loadtxt(xas_filename, usecols=(0, 1))
		#for row in raw_xas:
			#intensities[0][0][row[0]] = row[1]
	else:
		intensities[0][0] = -1

	return intensities


if __name__ == '__main__':
	atomLabels = np.empty((0, 1))
	intensities = np.empty((0, 1))

	snapshots = os.listdir(SNAPSHOTS_FOLDER_NAME)
	firstSnap = snapshots[0]
	firstSnapCoords = np.loadtxt(SNAPSHOTS_FOLDER_NAME + '/' + firstSnap, skiprows=2, usecols=(1, 2, 3))
	distanceArray = np.empty((0, len(firstSnapCoords)))
	calcDistanceNodeInst = CalcDistanceNode()

	for snap in snapshots:
		snap_path = SNAPSHOTS_FOLDER_NAME + '/' + snap
		currentSnapCoords = np.loadtxt(snap_path, skiprows=2, usecols=(1, 2, 3))
		currentSnapDistanceArray = calcDistanceNodeInst(currentSnapCoords, lattice_a, lattice_b, lattice_c, alpha, beta, gamma)
		distanceArray = np.vstack((distanceArray, currentSnapDistanceArray))
		
		currentSnapAtomLabels = np.loadtxt(snap_path, dtype=str, skiprows=2, usecols=(0,))
		currentSnapAtomLabels = np.reshape(currentSnapAtomLabels, (len(currentSnapAtomLabels), 1))
		atomLabels = np.vstack((atomLabels, currentSnapAtomLabels))

		currentSnapIntensities = np.empty((0, 1))
		for atomNum in range(len(currentSnapAtomLabels)):
			atomName = currentSnapAtomLabels[atomNum][0]
			atomIntensities = parse_intensities(atomName, atomNum+1, snap)
			currentSnapIntensities = np.vstack((currentSnapIntensities, atomIntensities))
			
		intensities = np.vstack((intensities, currentSnapIntensities))

	permuteNode = PermutationNode()
	permutedDistanceArray = permuteNode(distanceArray, atomLabels, intensities)

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
	
	np.set_printoptions(threshold='nan', precision=3)
	print permutedDistanceArray
