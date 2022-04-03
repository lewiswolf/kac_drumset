'''
This file contains the fixed internal types and classes used as part of this package.
'''

# core
import random

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from . import _geometry as _G
from . import geometry as G

__all__ = [
	'Polygon',
	'RandomPolygon',
]


class Polygon():
	'''
	A base class for a polygon, instantiated with an array of vertices.
	'''

	n: int								# number of vertices
	vertices: npt.NDArray[np.float64]	# cartesian products representing the corners of a shape

	def __init__(self, vertices: npt.NDArray[np.float64]) -> None:
		if (vertices.ndim != 2 or vertices.shape[1] != 2):
			raise ValueError('Array of vertices is not the correct shape: (n, 2)')
		self.vertices = vertices
		self.n = vertices.shape[0]


class RandomPolygon(Polygon):
	'''
	This class is used to generate a random polygon, normalised and centred between 0.0 and 1.0. The area and the centroid
	of the polygon are also included in this class.
	'''

	area: float							# area of the polygon
	centroid: tuple[float, float]		# coordinate pair representing the centroid of the polygon
	convex: bool						# is the polygon convex?
	mask: npt.NDArray[np.int8]			# discrete projection of the polygon, where 1 signifies the internal region of the shape

	def __init__(self, max_vertices: int, allow_concave: bool = True) -> None:
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
