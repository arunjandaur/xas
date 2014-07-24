class Peak():
	def __init__(self, peak_E, intensity, left, right):
		self._peak_E = peak_E
		self._intensity = intensity
		self._left = left
		self._right = right

	def getPeakEnergy(self):
		return self._peak_E

	def getIntensity(self):
		return self._intensity

	def getLeftEnergy(self):
		return self._left

	def getRightEnergy(self):
		return self._right

	def __str__(self):
		return "Energy: {0}, Intensity: {1}".format(self.getPeakEnergy(), self.getIntensity())
