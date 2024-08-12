'''
This file contains classes for generating polygons, including random polygon generators and unit polygon generators.
'''

# core
import random

# dependencies
import numpy as np 			# maths

# src
from ..externals._geometry import (
	_generateConvexPolygon,
	_generateIrregularStar,
	_generatePolygon,
	_generateUnitRectangle,
	# _generateUnitTriangle,
	_normaliseConvexPolygon,
	_normaliseSimplePolygon,
)
from .polygon import Polygon
from .types import ShapeSettings

__all__ = [
	'ConvexPolygon',
	'IrregularStar',
	'TravellingSalesmanPolygon',
	'UnitRectangle',
	# 'UnitTriangle',
]


class ConvexPolygon(Polygon):
	'''
	Generate convex shapes according to Pavel Valtr's 1995 algorithm.
	'''

	class Settings(ShapeSettings, total=False):
		''' Settings to be used when generating. '''
		N: int				# number of vertices (randomly generated when N < 3)
		max_vertices: int	# maximum number of vertices when generating

	def __init__(self, N: int = 0, max_vertices: int = 10) -> None:
		super().__init__(
			_normaliseConvexPolygon(_generateConvexPolygon(N if N > 2 else random.randint(3, max_vertices)), True),
		)


class IrregularStar(Polygon):
	'''
	This is a fast method for generating concave polygons, particularly with a large number of vertices. This approach
	generates polygons by ordering a series of random points around a centre point. As a result, not all possible simple
	polygons are generated this way.
	'''

	class Settings(ShapeSettings, total=False):
		''' Settings to be used when generating. '''
		N: int				# number of vertices (randomly generated when N < 3)
		max_vertices: int	# maximum number of vertices when generating

	def __init__(self, N: int = 0, max_vertices: int = 10) -> None:
		super().__init__(_generateIrregularStar(N if N > 2 else random.randint(3, max_vertices)))
		self.vertices = np.array(
			_normaliseConvexPolygon(self.vertices, True) if self.convex else _normaliseSimplePolygon(self.vertices, True),
		)


class TravellingSalesmanPolygon(Polygon):
	'''
	This algorithm is based on a method of eliminating self-intersections in a polygon by using the Lin and Kerningham
	'2-opt' moves. Such a move eliminates an intersection between two edges by reversing the order of the vertices between
	the edges. Intersecting edges are detected using a simple sweep through the vertices and then one intersection is
	chosen at random to eliminate after each sweep.
	van Leeuwen, J., & Schoone, A. A. (1982). Untangling a traveling salesman tour in the plane.
	'''

	class Settings(ShapeSettings, total=False):
		''' Settings to be used when generating. '''
		N: int				# number of vertices (randomly generated when N < 3)
		max_vertices: int	# maximum number of vertices when generating

	def __init__(self, N: int = 0, max_vertices: int = 10) -> None:
		super().__init__(_generatePolygon(N if N > 2 else random.randint(3, max_vertices)))
		self.vertices = np.array(
			_normaliseConvexPolygon(self.vertices, True) if self.convex else _normaliseSimplePolygon(self.vertices, True),
		)


class UnitRectangle(Polygon):
	'''
	Define a rectangle with unit area and an aspect ration epsilon.
	'''

	epsilon: float

	class Settings(ShapeSettings, total=False):
		''' Settings to be used when generating. '''
		epsilon: float		# aspect ratio (randomly generated when epsilon = 0)

	def __init__(self, epsilon: float = 0.) -> None:
		self.epsilon = epsilon or np.random.uniform(0., 1.)
		super().__init__(_generateUnitRectangle(self.epsilon))

	def __getLabels__(self) -> dict[str, list[float | int]]:
		'''
		This method should be used to return the metadata about the current shape.
		'''
		return {'epsilon': [self.epsilon], 'N': [self.N], 'vertices': self.vertices.tolist()}


# class UnitTriangle(Polygon):
# 	'''
# 	Define a triangle with unit area. For any point (r, θ) where θ ∈ [0, π / 2] and r ∈ [0, 1], the corresponding
# 	triangle will be unique.
# 	'''

# 	r: float
# 	theta: float

# 	class Settings(ShapeSettings, total=False):
# 		''' Settings to be used when generating. '''
# 		r: float			# radius
# 		theta: float		# angle

# 	def __init__(self, r: float | None = None, theta: float | None = None) -> None:
# 		self.r = np.random.uniform(0., 1.) if r is None else r
# 		self.theta = np.random.uniform(0., np.pi) if theta is None else theta
# 		assert self.r <= 1. and self.r >= 0., 'r ∈ [0, 1]'
# 		super().__init__(_generateUnitTriangle(self.r, self.theta))

# 	def __getLabels__(self) -> dict[str, list[float | int]]:
# 		'''
# 		This method should be used to return the metadata about the current shape.
# 		'''
# 		return {'r': [self.r], 'N': [self.N], 'theta': [self.theta], 'vertices': self.vertices.tolist()}
