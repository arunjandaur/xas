import mdp

class AnglePermutationNode(mdp.Node):
	def is_trainable(self):
		return False

	def _execute(self, angles, atomLabels, intensities):
		
