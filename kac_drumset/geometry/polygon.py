'''
This file contains the fixed geometric types used as part of this package.
'''

# core
from typing import Optional, Union

# dependencies
import cv2					# image processing
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from ..externals._geometry import (
	_centroid,
	_isConvex,
	_isPointInsideConvexPolygon,
	_isSimple,
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
		vertices: Union[list[list[float]], npt.NDArray[np.float64]]

	def __init__(self, vertices: Optional[Union[list[list[float]], npt.NDArray[np.float64]]] = None) -> None:
		'''
		input:
			vertices = array of cartesian points.
		'''
		assert vertices is not None
		self.vertices = np.array(vertices)
		assert self.vertices.ndim == 2 and self.vertices.shape[1] == 2, \
			'Array of vertices is not the correct shape: (n, 2)'
		self.N = self.vertices.shape[0]
		assert self.N >= 3, 'A polygon must have three vertices.'
		self.convex = _isConvex(vertices)

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
		An implementation of the shoelace algorithm, first described by Albrecht Ludwig Friedrich Meister, which is used to
		calculate the area of a polygon.
		'''
		return _polygonArea(self.vertices)

	def centroid(self) -> tuple[float, float]:
		'''
		This algorithm is used to calculate the geometric centroid of a 2D polygon.
		See http://paulbourke.net/geometry/polygonmesh/ 'Calculating the area and centroid of a polygon'.
		'''
		out = _centroid(self.vertices, self.area())
		return out[0], out[1]

	def draw(self, grid_size: int) -> npt.NDArray[np.int8]:
		'''
		This function creates a boolean mask of a polygon on a grid with dimensions R^(grid_size). The input shape should
		exist within a domain R^G where G ∈ [0, 1].
		'''
		# transposing maintains that mask[x, y] works as intended
		return cv2.fillConvexPoly(
			np.zeros((grid_size, grid_size), 'int8'),
			np.array([[round(y * (grid_size - 1)), round(x * (grid_size - 1))] for [x, y] in self.vertices], 'int32'),
			1,
		) if self.convex else cv2.fillPoly(
			np.zeros((grid_size, grid_size), 'int8'),
			np.array([[[round(y * (grid_size - 1)), round(x * (grid_size - 1))] for [x, y] in self.vertices]], 'int32'),
			1,
		)

	def isPointInside(self, p: tuple[float, float]) -> bool:
		'''
		Determines whether or not a cartesian point is within a polygon, including boundaries.
		'''
		assert self.convex, 'isPointInsidePolygon() does not currently support concave shapes.'
		return _isPointInsideConvexPolygon(list(p), self.vertices)

	def isSimple(self) -> bool:
		'''
		Determine if a polygon is simple by checking for intersections.
		'''
		return _isSimple(self.vertices)
