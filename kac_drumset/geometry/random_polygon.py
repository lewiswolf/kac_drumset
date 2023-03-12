'''
RandomPolygon is used to generate a random drum shape alongside computing some of its properties.
'''

# core
import random

# src
from ._generate_polygon import generatePolygon, generateConvexPolygon
from ._morphisms import normaliseConvexPolygon, normalisePolygon
from ._polygon_properties import centroid, isConvex
from .types import Polygon

__all__ = [
	'RandomPolygon',
]


class RandomPolygon(Polygon):
	'''
	This class is used to generate a random polygon, normalised and centred between 0.0 and 1.0. The convexity and the
	centroid of the polygon are also included in this class.
	'''

	centroid: tuple[float, float]		# coordinate pair representing the centroid of the polygon

	def __init__(self, max_vertices: int, allow_concave: bool = False) -> None:
		'''
		This function generates a polygon, whilst also calculating its properties.
		input:
			max_vertices:	Maximum amount of vertices. The true value is a uniform distribution from 3 to max_vertices.
			allow_concave:	Is this polygon allowed to be concave?
		'''

		# generate random polygon
		if not allow_concave or random.getrandbits(1):
			super().__init__(generateConvexPolygon(random.randint(3, max_vertices)))
		else:
			super().__init__(generatePolygon(random.randint(3, max_vertices)))
		# normalise
		self.vertices = normaliseConvexPolygon(self) if isConvex(self) else normalisePolygon(self)
		# calculate other properties
		self.centroid = centroid(self)
