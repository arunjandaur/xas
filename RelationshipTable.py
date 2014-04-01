class RelationshipTable():
	def __init__(self, coordType):
		self.coordType = coordType
		self.depth = 0
		_build_dictionary()

	def _build_dictionary(self):
		if self.coordType == 'distance':
			self.depth = 2

		if self.coordType == 'angle':
			self.depth = 3

		if self.coordType == 'dihedral':
			self.depth = 4

	def get_coord_type(self):
		return self.coordType

	def 
