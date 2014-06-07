class Peak():
	def __init__(self, peak_E, intensity):
		self._peak_E = peak_E
		self._intensity = intensity

	def getPeakEnergy(self):
		return self._peak_E

	def getIntensity(self):
		return self._intensity

	def __str__(self):
		return "Energy: {0}, Intensity: {1}".format(self.getPeakEnergy(), self.getIntensity())
