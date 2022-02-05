# core
import random

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from . import geometry as g


class RandomPolygon():
	'''
	This class is used to generate a random polygon, normalised and centred between 0.0
	and 1.0. Various properties relating to this polygon are also attached to this class.
	'''

	area: float							# area of the polygon
	centroid: tuple[float, float]		# coordinate pair representing the centroid of the polygon
	convex: bool						# is the polygon convex?
	mask: npt.NDArray[np.int8]			# discrete projection of the polygon, where 1 signifies the internal region of the shape
	n: int								# the number of vertices
	vertices: npt.NDArray[np.float64]	# cartesian products representing the corners of a shape

	def __init__(self, max_vertices: int, grid_size: int = 0, allow_concave: bool = True) -> None:
		'''
		This function generates a polygon, whilst also calculating its properties.
		params:
			max_vertices:	Maximum amount of vertices. The true value is a uniform
							distribution from 3 to max_vertices.
			grid_size:		Every polygon has a boolean mask as one of its properties.
							This variable controls the size of that mask.
			allow_concave:	Is this polygon allowed to be concave?
		'''

		# generate random polygon
		self.n = random.randint(3, max_vertices)
		if not allow_concave or random.getrandbits(1):
			self.vertices = g.generateConvex(self.n)
			self.convex = True
		else:
			self.vertices = g.generateConcave(self.n)
			self.convex = g.isConvex(self.vertices)

		# normalise
		self.vertices = g.groupNormalisation(self.vertices, convex=self.convex)

		# calculate other properties
		self.area = g.area(self.vertices)
		self.centroid = g.centroid(self.vertices, self.area)

		# compute the boolean mask
		if grid_size:
			self.mask = g.booleanMask(self.vertices, grid_size, convex=self.convex)
