import numpy as np
import os

from CalcDistanceNode import CalcDistanceNode
from CalcAngleNode import CalcAngleNode
from PermutationNode import PermutationNode
from parse_intensities import parse_intensities
from peak_shift_analysis import sum_gaussians_fit
from peak_tracking import SA, linreg
from scipy.stats import pearsonr
from sklearn.mixture import DPGMM

SNAPSHOTS_FOLDER_NAME = "snapshots"
OUTPUT_FOLDER = "dist_and_intens"
lattice_a = 32 #8.99341
lattice_b = 32 #8.99341
lattice_c = 32 #8.99341
alpha = 90.0
beta = 90.0
gamma = 90.0

def parseXYZ_Intens(snap, excited_atom):
    intens = np.empty((0, 1000), dtype=('f8, f8'))
    
    #Atom Labels
    snap_path = SNAPSHOTS_FOLDER_NAME + '/' + snap
    atomLabels = np.loadtxt(snap_path, dtype=str, skiprows=2, usecols=(0,)) #Parse atoms labels and stack them
    atomLabels = np.reshape(atomLabels, (len(atomLabels), 1))
    
    #Coords
    coords = np.loadtxt(snap_path, skiprows=2, usecols=(1, 2, 3))
    
    #Intensities
    for atomNum in range(len(atomLabels)): #For each atom, parse its intens
        atomName = atomLabels[atomNum][0]
        if atomName == excited_atom:
            atomIntens = parse_intensities(atomName, atomNum+1, snap)
            intens = np.vstack((intens, atomIntens)) #Accumulate intens of all atoms in this snap
    
    return atomLabels, coords, intens

def extractData(coords, atomLabels, excited_atom, periodic, radius=100):
    atomSet = list(set(np.reshape(atomLabels, (len(atomLabels),)).tolist()))
    atomSet.sort()

    distNode = CalcDistanceNode()
    if periodic == True:
        dists = distNode(coords, lattice_a, lattice_b, lattice_c, alpha, beta, gamma)
    else:
        dists = distNode(coords)
    inputData = []
    for i in range(len(atomLabels)):
        currAtom = atomLabels[i]
        currCoord = coords[i]
        if currAtom == excited_atom:
            input_row_data = []
            done = []
            for label in atomSet:
                for label2 in atomSet:
                    if (label2 + label) not in done or label2 == label:
                        tempDists = []
                        for j in range(len(coords)):
                            atom1 = atomLabels[j]
                            atom1Coord = coords[j]
                            for k in range(len(coords)):
                                atom2 = atomLabels[k]
                                atom2Coord = coords[k]
                                if k != j and atom1 == label and atom2 == label2:
                                    curr_atom1_dist = dists[i][j]
                                    curr_atom2_dist = dists[i][k]
                                    if curr_atom1_dist < radius and curr_atom2_dist < radius:
                                        atom1_2_dist = dists[j][k]
                                        if dists[k][j] not in tempDists:
                                            tempDists.append(atom1_2_dist)
                        tempDists.sort()
                        input_row_data.extend(tempDists)
                        done.append(label + label2)
            inputData.append(input_row_data)
    
    angleNode = CalcAngleNode()
    if periodic == True:
        angles = angleNode(coords, lattice_a, lattice_b, lattice_c, alpha, beta, gamma)
    else:
        angles = angleNode(coords)
    angleData = []
    for n in range(len(atomLabels)):
        currAtom = atomLabels[n]
        currCoord = coords[n]
        if currAtom == excited_atom:
            input_row_data = []
            done = []
            for label in atomSet:
                for label2 in atomSet:
                    for label3 in atomSet:
                        tempAngles = []
                        for i in range(len(coords)):
                            if atomLabels[i] == label:
                                for j in range(len(coords)):
                                    if atomLabels[j] == label2:
                                        for k in range(len(coords)):
                                            if atomLabels[k] == label3:
                                                if i != j and j != k and i != k:
                                                    curr_atom1_dist = dists[n][i]
                                                    curr_atom2_dist = dists[n][j]
                                                    curr_atom3_dist = dists[n][k]
                                                    if max(curr_atom1_dist, curr_atom2_dist, curr_atom3_dist) < radius:
                                                        if angles[i][k][j] != np.inf:
                                                            tempAngles.append(angles[i][j][k])
                                                            angles[i][j][k] = np.inf
                        tempAngles.sort()
                        input_row_data.extend(tempAngles)
            angleData.append(input_row_data)

    inputData = np.hstack((np.array(inputData), np.array(angleData)))
    return inputData

def splitPeakData(peakData):
    return peakData[:, :-1], np.array([peakData[:, -1]]).T

def preparePeakData(inputData, intensities):
    peakData = []
    for i in range(len(intensities)):
        num_i, amps_i, means_i, sigmas_i = sum_gaussians_fit(energies, intensities[i])
        for peak in means_i:
            peakData.append(np.hstack((inputData[i], mean)))
    peakData = np.array(peakData)
    return splitPeakData(peakData)

def cluster(inputData, peaks):
    dpgmm = DPGMM(10, n_iter=1000)
    dpgmm.fit(np.hstack((inputData, peaks)))
    print dpgmm.weights_
    print dpgmm.means_

def linearity(inputData, peaks):
    pearsonr(inputData, peaks)

if __name__ == "__main__":
    excited_atom = "C"
    radius = 15 #Angstroms
    snapshots = os.listdir(SNAPSHOTS_FOLDER_NAME)

    atomLabels, coords, intens = parseXYZ_Intens(snapshots[0], excited_atom)
    inputData = extractData(coords, atomLabels, excited_atom, radius)
    for snap in snapshots[1:]:
        currAtomLabels, currCoords, currIntens = parseXYZ_Intens(snap, excited_atom)
        currInputData = extractData(currCoords, currAtomLabels, excited_atom, False, radius)
        inputData = np.vstack((inputData, currInputData))
        intens = np.vstack((intens, currIntens))
    print inputData

    energies = []
    intensities = []
    for E_I in intens[0]:
        energies.append(E_I[0])
        intensities.append(E_I[1])
    energies = np.array(energies)
    intensities = np.array(intensities)
    for row in intens[1:]:
        rowIntens = []
        for E_I in row:
            rowIntens.append(E_I[1])
        intensities = np.vstack((intensities, np.array(rowIntens)))

    inputData, peaks = preparePeakData(inputData, intensities)
    clusters = cluster(inputData, peaks)

    for cluster in clusters:
        inputData = cluster[0]
        peaks = cluster[1]
        if abs(linearity(inputData, peaks)) > .8:
            error, params = linreg(cluster)
        else:
            config, error, params = SA(inputData, peaks)

##########

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
        for atomNum in range(len(currSnapAtomLabels)): #For each atom, parse its intens
            atomName = currSnapAtomLabels[atomNum][0]
            atomIntens = parse_intensities(atomName, atomNum+1, snap)
            currSnapIntens = np.vstack((currSnapIntens, atomIntens)) #Accumulate intens of all atoms in this snap
        intens = np.vstack((intens, currSnapIntens)) #Accumulate the intens of all snaps
        #Intensities
    
    #Distances
    permuteNode = PermutationNode()
    #Nested dict; Outer keys: atoms, Outer values: (dicts, atom intens)
    #Inner keys: atoms, Inner values: Distance matrix between inner and outer key
    distanceMaster = permuteNode(distArr, atomLabels, intens)
    #Distances

    return distanceMaster, angleMaster, intens
