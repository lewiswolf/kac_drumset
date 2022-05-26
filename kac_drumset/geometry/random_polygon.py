'''
RandomPolygon is used to generate a random drum shape alongside computing some of its properties.
'''

# core
import random

# src
from . import _geometry as _G
from . import geometry as G
from .types import Polygon

__all__ = [
	'RandomPolygon',
]


class RandomPolygon(Polygon):
	'''
	This class is used to generate a random polygon, normalised and centred between 0.0 and 1.0. The area and the centroid
	of the polygon are also included in this class.
	'''

	area: float							# area of the polygon
	centroid: tuple[float, float]		# coordinate pair representing the centroid of the polygon
	convex: bool						# is the polygon convex?

	def __init__(self, max_vertices: int, allow_concave: bool = False) -> None:
		'''
		This function generates a polygon, whilst also calculating its properties.
		input:
			max_vertices:	Maximum amount of vertices. The true value is a uniform distribution from 3 to max_vertices.
			allow_concave:	Is this polygon allowed to be concave?
		'''

		# generate random polygon
		if not allow_concave or random.getrandbits(1):
			super().__init__(_G.generateConvexPolygon(random.randint(3, max_vertices)))
			self.convex = True
		else:
			super().__init__(G.generateConcave(random.randint(3, max_vertices)))
			self.convex = _G.isConvex(self.vertices)

		# normalise
		self.vertices = G.groupNormalisation(self.vertices, convex=self.convex)

		# calculate other properties
		self.area = G.area(self.vertices)
		self.centroid = G.centroid(self.vertices, self.area)
