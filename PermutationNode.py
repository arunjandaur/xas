import mdp
import numpy as np

class PermutationNode(mdp.Node):
	def is_trainable(self):
		return False

	def array_to_list(self, numpy_array):
		"""
		Order MUST be preserved
		Input:
			numpy_array: 2D array
		Output:
			A python array that is the concatenation of all the rows of the input 2D array
		"""
		numpy_array = numpy_array.tolist()
		array = []
		for subList in numpy_array:
			array.extend(subList)
		return array

	def split(self, distanceArray, atomLabels, atomLabelsSet, intensities):
		"""
		Input:
			distanceArray: distance matrix such that the (i, j)th element is the dist(atom i, atom j)
			atomLabels: A vertical 2D array where each row has one element: the atom label such that the row index corresponds to the line number (of the .xyz file) the atom was found on.
			atomLabelsSet: A set of the atomLabels (no repeats)
			intensities: A 2D array where each row has one element: a dictionary of 1000 values. Each key is one of the 1000 energy levels and the corresponding value is the intensity that a specific atom emits at that energy. Each row corresponds to a specific atom (row index = line index atom was found on in the .xyz file).
		Output:
			A dictionary whose keys are atoms and whose values are arrays that contain the intensities of the atom and the distance matrix between the atom and all other atoms.
		"""
		master = {}
		for atom in atomLabelsSet: #For each unique atom label (C, N, O, etc)
			subDistArray = np.empty((0, len(distanceArray[0])))
			subIntensities = np.empty((0, 1))
			for row in range(len(distanceArray)):
				if atomLabels[row][0] == atom:
					subDistArray = np.vstack((subDistArray, distanceArray[row])) #Build distance array
					subIntensities = np.vstack((subIntensities, intensities[row])) #Build intensity array
			master[atom] = [subDistArray, subIntensities] #Add the atom as a key whose value is an array containing the distances and intensities associated with that atom. Repeat this for all atoms.
		return master

	def subsplit(self, distanceArray, atomLabels, atomLabelsSet):
		"""
		This does almost the same thing as split. It does not do any intensity splitting. Instead, it just splits the distances.
		"""
		master = {}
		for atom in atomLabelsSet:
			subDistArray = np.empty((0, len(distanceArray[0])))
			for row in range(len(distanceArray)):
				if atomLabels[row][0] == atom:
					subDistArray = np.vstack((subDistArray, distanceArray[row]))
			master[atom] = subDistArray
		return master

	def _execute(self, distanceArray, atomLabels, intensities):
		atomLabelsSet = set(self.array_to_list(atomLabels))
		master = self.split(distanceArray, atomLabels, atomLabelsSet, intensities)
		for atom in master:
			#For each atom in master, split the the internal distance matrices as well using subsplit. Do not split intensities because they are associated with the outer atoms. Now each atom's value in the key value pair does not contain a distance matrix in the first element of the array, but instead a dictionary where each key is the atom that the outer atom is connected to and the values are the distance matrices between the outer atom key and the inner atom key.
			master[atom][0] = self.subsplit(np.transpose(master[atom][0]), atomLabels, atomLabelsSet)
			for atom2 in master[atom][0]:
				master[atom][0][atom2] = np.transpose(master[atom][0][atom2])

		for atom in master:
			#Now sort the distances to roughly cluster similar distances together
			for atom2 in master[atom][0]:
				subArray = master[atom][0][atom2]
				subArray.sort()
	
		return master
