import mdp
import numpy as np

class PermutationNode(mdp.Node):
	def is_trainable(self):
		return False

	def array_to_list(self, numpy_array):
		"""
		Order MUST be preserved
		"""
		numpy_array = numpy_array.tolist()
		array = []
		for subList in numpy_array:
			array.extend(subList)
		return array

	def split(self, distanceArray, atomLabels, atomLabelsSet, intensities):
		master = {}
		for atom in atomLabelsSet:
			subDistArray = np.empty((0, len(distanceArray[0])))
			subIntensities = np.empty((0, 1))
			for row in range(len(distanceArray)):
				if atomLabels[row][0] == atom:
					subDistArray = np.vstack((subDistArray, distanceArray[row]))
					subIntensities = np.vstack((subIntensities, intensities[row]))
			master[atom] = [subDistArray, subIntensities]
		return master

	def subsplit(self, distanceArray, atomLabels, atomLabelsSet):
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
			master[atom][0] = self.subsplit(np.transpose(master[atom][0]), atomLabels, atomLabelsSet)
			for atom2 in master[atom][0]:
				master[atom][0][atom2] = np.transpose(master[atom][0][atom2])

		for atom in master:
			for atom2 in master[atom][0]:
				subArray = master[atom][0][atom2][0]
				subArray.sort()
	
		return master
