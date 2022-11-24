'''
This file contains the fixed geometric types used as part of this package.
'''

# core
from abc import ABC, abstractmethod

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from ..externals._geometry import _polygonArea

__all__ = [
	'Polygon',
	'Shape',
]


class Shape(ABC):
	'''
	An abstract base class for a shape in Euclidean geometry.
	'''

	def __init__(self) -> None:
		pass

	@abstractmethod
	def area(self) -> float:
		pass


class Polygon(Shape):
	'''
	A base class for a polygon, instantiated with an array of vertices.
	'''

	N: int								# number of vertices
	vertices: npt.NDArray[np.float64]	# cartesian products representing the vertices of a shape

	def __init__(self, vertices: npt.NDArray[np.float64]) -> None:
		assert vertices.ndim == 2 and vertices.shape[1] == 2, \
			'Array of vertices is not the correct shape: (n, 2)'
		self.vertices = vertices
		self.N = vertices.shape[0]

	def area(self) -> float:
		'''
		An implementation of the shoelace algorithm, first described by Albrecht Ludwig Friedrich Meister, which is used to
		calculate the area of a polygon.
		'''
		return _polygonArea(self.vertices.tolist())
