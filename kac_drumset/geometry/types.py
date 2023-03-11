'''
This file contains the fixed geometric types used as part of this package.
'''

# core
from abc import ABC, abstractmethod
from typing import Union

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from ..externals._geometry import _isConvex, _polygonArea

__all__ = [
	'Circle',
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


class Circle(Shape):
	'''
	A base class for a circle, instantiated with a radius.
	'''

	r: float 							# radius

	def __init__(self, r: float = 1.) -> None:
		'''
		input:
			r = radius
		'''
		self.r = r

	def area(self) -> float:
		''' Archimedes. '''
		return np.pi * (self.r ** 2.)


class Polygon(Shape):
	'''
	A base class for a polygon, instantiated with an array of vertices.
	'''

	N: int								# number of vertices
	vertices: npt.NDArray[np.float64]	# cartesian products representing the vertices of a shape

	def __init__(self, vertices: Union[list[list[float]], npt.NDArray[np.float64]]) -> None:
		'''
		input:
			vertices = array of cartesian points.
		'''
		self.vertices = np.array(vertices)
		assert self.vertices.ndim == 2 and self.vertices.shape[1] == 2, \
			'Array of vertices is not the correct shape: (n, 2)'
		self.N = self.vertices.shape[0]
		assert self.N >= 3, 'A polygon must have three vertices.'

	def area(self) -> float:
		'''
		An implementation of the shoelace algorithm, first described by Albrecht Ludwig Friedrich Meister, which is used to
		calculate the area of a polygon.
		'''
		return _polygonArea(self.vertices)

	def convex(self) -> bool:
		'''
		Tests whether or not a given polygon is convex. This is achieved using the resultant sign of the cross product for
		each vertex:
			[(x_i - x_i-1), (y_i - y_i-1)] × [(x_i+1 - x_i), (y_i+1 - y_i)].
		See => http://paulbourke.net/geometry/polygonmesh/ 'Determining whether or not a polygon (2D) has its vertices ordered
		clockwise or counter-clockwise'.
		'''
		return _isConvex(self.vertices)
