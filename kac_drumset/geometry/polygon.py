'''
This file contains the fixed geometric types used as part of this package.
'''

# core
from typing import cast, Optional, Union

# dependencies
import cv2					# image processing
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from ..externals._geometry import (
	_centroid,
	_isConvex,
	_isPointInsideConvexPolygon,
	_isPointInsidePolygon,
	_isSimple,
	_normalisePolygon,
	_polygonArea,
)
from .types import Shape, ShapeSettings

__all__ = [
	'Polygon',
]


class Polygon(Shape):
	'''
	A base class for a polygon, instantiated with an array of vertices.
	'''

	N: int								# number of vertices
	_convex: bool						# is the polygon convex or not?
	_vertices: npt.NDArray[np.float64]	# cartesian products representing the vertices of a shape

	class Settings(ShapeSettings, total=False):
		''' Settings to be used when generating. '''
		vertices: Union[list[list[float]], npt.NDArray[np.float64]]

	def __init__(self, vertices: Optional[Union[list[list[float]], npt.NDArray[np.float64]]] = None) -> None:
		self.vertices = np.array(vertices)
		self.N = self.vertices.shape[0]
		assert vertices is not None, 'Polygon() must be initialised with an array of vertices.'
		assert self.vertices.ndim == 2 and self.vertices.shape[1] == 2, 'Array of vertices is not the correct shape: (n, 2)'
		assert self.N >= 3, 'A polygon must have three vertices.'

	'''
	Getters and setters for .convex and .vertices.
	This maintains that .convex is a cached variable, but is also updated with the vertices.
	'''

	@property
	def convex(self) -> bool:
		return self._convex

	@convex.setter
	def convex(self, v: bool) -> None:
		pass

	@property
	def vertices(self) -> npt.NDArray[np.float64]:
		return self._vertices

	@vertices.setter
	def vertices(self, v: npt.NDArray[np.float64]) -> None:
		self._vertices = v
		self._convex = _isConvex(v)

	def area(self) -> float:
		'''
		An implementation of the polygon area algorithm derived using Green's Theorem.
		https://math.blogoverflow.com/2014/06/04/greens-theorem-and-area-of-polygons/
		'''
		return _polygonArea(self.vertices)

	def centroid(self) -> tuple[float, float]:
		'''
		This algorithm is used to calculate the geometric centroid of a 2D polygon.
		See http://paulbourke.net/geometry/polygonmesh/ 'Calculating the area and centroid of a polygon'.
		'''
		return cast(tuple[float, float], tuple(_centroid(self.vertices, self.area())))

	def draw(self, grid_size: int) -> npt.NDArray[np.int8]:
		'''
		This function creates a boolean mask of a manifold on a grid with dimensions R^(grid_size). The input shape is always
		normalised to the domain R^G before being drawn.
		'''
		# transposing maintains that mask[x, y] works as intended
		return cv2.fillConvexPoly(
			np.zeros((grid_size, grid_size), 'int8'),
			np.array(
				[
					[round(y * (grid_size - 1)), round(x * (grid_size - 1))]
					for [x, y] in (
						self.vertices if self.vertices.min() == 0. and self.vertices.max() == 1. else _normalisePolygon(self.vertices)
					)
				],
				'int32',
			),
			1,
		) if self.convex else cv2.fillPoly(
			np.zeros((grid_size, grid_size), 'int8'),
			np.array(
				[
					[round(y * (grid_size - 1)), round(x * (grid_size - 1))]
					for [x, y] in (
						self.vertices if self.vertices.min() == 0. and self.vertices.max() == 1. else _normalisePolygon(self.vertices)
					)
				],
				'int32',
			),
			1,
		)

	def isPointInside(self, p: tuple[float, float]) -> bool:
		'''
		Determines if a given point p ∈ P, including boundaries.
		'''
		return _isPointInsideConvexPolygon(
			list(p),
			self.vertices,
		) if self.convex else _isPointInsidePolygon(
			list(p),
			self.vertices,
		)

	def isSimple(self) -> bool:
		'''
		Determine if a polygon is simple by checking for intersections.
		'''
		return _isSimple(self.vertices)
