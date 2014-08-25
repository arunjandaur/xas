import numpy as np
import math
import os
import sys

from CalcDistanceNode import CalcDistanceNode
from CalcAngleNode import CalcAngleNode
from parse_intensities import parse_intensities
from peak_shift_analysis import sum_gaussians_fit
from peak_tracking import SA, linreg
from scipy.stats import pearsonr
from sklearn.mixture import DPGMM

def parseXYZ_Intens(snap, excited_atom, snapshots_folder, xas_folder):
    intens = np.empty((0, 1000), dtype=('f8, f8'))
    
    #Atom Labels
    snap_path = snapshots_folder + '/' + snap
    atomLabels = np.loadtxt(snap_path, dtype=str, skiprows=2, usecols=(0,)) #Parse atoms labels and stack them
    atomLabels = np.reshape(atomLabels, (len(atomLabels), 1))
    
    #Coords
    coords = np.loadtxt(snap_path, skiprows=2, usecols=(1, 2, 3))
    
    #Intensities
    for atomNum in range(len(atomLabels)): #For each atom, parse its intens
        atomName = atomLabels[atomNum][0]
        if atomName == excited_atom:
            atomIntens = parse_intensities(atomName, atomNum+1, snap, xas_folder)
            intens = np.vstack((intens, atomIntens)) #Accumulate intens of all atoms in this snap
    
    return atomLabels, coords, intens

def genDistRelations(coords, dists, atomLabels, atomSet, excited_atom, radius):
    distInputData = []
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
            distInputData.append(input_row_data)
    return np.array(distInputData)

def genAngleRelations(coords, dists, angles, atomLabels, atomSet, excited_atom, radius):
    angleInputData = []
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
            angleInputData.append(input_row_data)
    return np.array(angleInputData)

def extractData(coords, atomLabels, excited_atom, radius, periodic=False, *args):
    atomSet = list(set(np.reshape(atomLabels, (len(atomLabels),)).tolist()))
    atomSet.sort()

    distNode = CalcDistanceNode()
    angleNode = CalcAngleNode()
    if periodic == True:
        dists = distNode(coords, lattice_a, lattice_b, lattice_c, alpha, beta, gamma)
        angles = angleNode(coords, lattice_a, lattice_b, lattice_c, alpha, beta, gamma)
    else:
        dists = distNode(coords)
        angles = angleNode(coords)

    distInputData = genDistRelations(coords, dists, atomLabels, atomSet, excited_atom, radius)
    angleInputData = genAngleRelations(coords, dists, angles, atomLabels, atomSet, excited_atom, radius)
    return np.hstack((distInputData, angleInputData))

def splitPeakData(peakData):
    return peakData[:, :-1], np.array([peakData[:, -1]]).T

def preparePeakData(inputData, intensities):
    peakData = []
    num_peaks = []
    for i in range(len(intensities)):
        num_i, amps_i, means_i, sigmas_i = sum_gaussians_fit(energies, intensities[i])
        num_peaks.append(len(means_i))
        for peak in means_i:
            peakData.append(np.hstack((inputData[i], mean)))
    peakData = np.array(peakData)
    inputData, peaks = splitPeakData(peakData)
    return inputData, peaks, math.ceil(sum(num_peaks) / len(num_peaks))

def cluster(inputData, peaks):
    dpgmm = DPGMM(10, n_iter=1000)
    dpgmm.fit(np.hstack((inputData, peaks)))
    print dpgmm.weights_
    print dpgmm.means_

def linearity(inputData, peaks):
    pearsonr(inputData, peaks)

if __name__ == "__main__":
    args = sys.argv
    excited_atom, snapshots_folder, xas_folder, radius, periodic = args[1:6]
    snapshots = os.listdir(snapshots_folder)
    if periodic == "true":
        unit_cell, lattice_a, lattice_b, lattice_c, alpha, beta, gamma = [float(arg) for arg in args[6:]]
        periodic = True
    else:
        periodic = False

    atomLabels, coords, intens = parseXYZ_Intens(snapshots[0], excited_atom, snapshots_folder, xas_folder)
    if periodic == False:
        inputData = extractData(coords, atomLabels, excited_atom, radius)
    else:
        inputData = extractData(coords, atomLabels, excited_atom, radius, periodic, lattice_a, lattice_b, lattice_c, alpha, beta, gamma)
    for snap in snapshots[1:]:
        currAtomLabels, currCoords, currIntens = parseXYZ_Intens(snap, excited_atom, snapshots_folder, xas_folder)
        if periodic == False:
            currInputData = extractData(currCoords, currAtomLabels, excited_atom, radius)
        else:
            currInputData = extractData(currCoords, currAtomLabels, excited_atom, radius, periodic, lattice_a, lattice_b, lattice_c, alpha, beta, gamma)
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

    inputData, peaks, avg_num_peaks = preparePeakData(inputData, intensities)
    clusters = cluster(inputData, peaks)

    for cluster in clusters:
        inputData = cluster[0]
        peaks = cluster[1]
        if abs(linearity(inputData, peaks)) > .8:
            error, params = linreg(cluster)
        else:
            config, error, params = SA(inputData, peaks)

##########

def expand(matr):
    matr = np.transpose(matr)
    d1 = matr[0]
    d2 = matr[1]
    a = matr[2]
    cosa = np.cos(a)
    sina = np.sin(a)
    matr = np.transpose(np.vstack((d1, d2, a)))
    return matr
