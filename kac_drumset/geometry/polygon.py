'''
This file contains the fixed geometric type Polygon.
'''

# core
from typing import cast

# dependencies
import cv2					# image processing
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from ..externals._geometry import (
	_isConvex,
	_isPointInsideConvexPolygon,
	_isPointInsidePolygon,
	_isSimple,
	_normalisePolygon,
	_polygonArea,
	_polygonCentroid,
)
from .types import Shape, ShapeSettings

__all__ = [
	'Polygon',
]


class Polygon(Shape):
	'''
	A base class for a polygon, instantiated with an array of vertices.
	'''

	_convex: bool						# is the polygon convex or not?
	_vertices: npt.NDArray[np.float64]	# cartesian products representing the vertices of a shape

	class Settings(ShapeSettings, total=False):
		''' Settings to be used when generating. '''
		vertices: list[list[float]] | npt.NDArray[np.float64]

	def __init__(self, vertices: list[list[float]] | npt.NDArray[np.float64]) -> None:
		self.vertices = np.array(vertices)

	'''
	Getters and setters for area.
	Setting area _should_ be used to scale the polygon, but is not currently implemented.
	'''

	@property
	def area(self) -> float:
		''' An implementation of the polygon area algorithm derived using Green's Theorem. '''
		return _polygonArea(self.vertices)

	@area.setter
	def area(self, a: float) -> None:
		pass

	'''
	Getters and setters for centroid. Setting centroid translates the polygon about the plane.
	'''

	@property
	def centroid(self) -> tuple[float, float]:
		''' This algorithm is used to calculate the geometric centroid of a 2D polygon. '''
		return cast(tuple[float, float], tuple(_polygonCentroid(self.vertices)))

	@centroid.setter
	def centroid(self, c: tuple[float, float]) -> None:
		centroid = self.centroid
		self._vertices[:, 0] += c[0] - centroid[0]
		self._vertices[:, 1] += c[1] - centroid[1]

	'''
	Getters and setters for convex and vertices. This setup maintains that convex is a cached variable, that updates
	whenever the vertices are updated.
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
		assert self.vertices.ndim == 2 and self.vertices.shape[1] == 2, 'Array of vertices is not the correct shape: (n, 2)'
		assert self.N >= 3, 'A polygon must have three vertices.'

	'''
	Getters and setters for N. This approach maintains that N is immutable.
	'''

	@property
	def N(self) -> int:
		return self.vertices.shape[0]

	@N.setter
	def N(self) -> None:
		pass

	def __getLabels__(self) -> dict[str, list[float | int]]:
		'''
		This method should be used to return the metadata about the current shape.
		'''
		return {'N': [self.N], 'vertices': self.vertices.tolist()}

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
						self.vertices if self.vertices.min() == 0. and self.vertices.max() == 1. else _normalisePolygon(
							self.vertices,
							False,
						)
					)
				],
				'int32',
			),
			(1, 0, 0),
		).astype(np.int8) if self.convex else cv2.fillPoly(
			np.zeros((grid_size, grid_size), 'int8'),
			[np.array([
				[round(y * (grid_size - 1)), round(x * (grid_size - 1))]
				for [x, y] in (
					self.vertices if self.vertices.min() == 0. and self.vertices.max() == 1. else _normalisePolygon(
						self.vertices,
						False,
					)
				)
			])],
			(1, 0, 0),
		).astype(np.int8)

	def isPointInside(self, p: tuple[float, float]) -> bool:
		'''
		Determines if a given point p ∈ P, including boundaries.
		'''
		return _isPointInsideConvexPolygon(p, self.vertices) if self.convex else _isPointInsidePolygon(p, self.vertices)

	def isSimple(self) -> bool:
		'''
		Determine if a polygon is simple by checking for intersections.
		'''
		return _isSimple(self.vertices)
