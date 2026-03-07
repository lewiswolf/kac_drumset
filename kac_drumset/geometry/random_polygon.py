'''
This file contains classes for generating polygons, including random polygon generators and unit polygon generators.
'''

# core
import math
import random

# dependencies
import numpy as np 			# maths

# src
from ..externals._geometry import (
	_generateConvexPolygon,
	_generateIrregularStar,
	_generateRegularPolygon,
	_generateRegularStar,
	_generatePolygon,
	_generateUnitRectangle,
	_generateUnitTriangle,
	_normaliseConvexPolygon,
	_normaliseSimplePolygon,
)
from .polygon import Polygon
from .types import ShapeSettings

__all__ = [
	'ConvexPolygon',
	'IrregularStar',
	'RegularPolygon',
	'RegularStar',
	'TravellingSalesmanPolygon',
	'UnitRectangle',
	'UnitTriangle',
]


class ConvexPolygon(Polygon):
	'''
	Generate convex shapes according to Pavel Valtr's 1995 algorithm.
	Adapted from Sander Verdonschot's Java version, found here:
	https://cglab.ca/~sander/misc/ConvexGeneration/ValtrAlgorithm.java
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
			_normaliseConvexPolygon(self.vertices, True) if self.convex() else _normaliseSimplePolygon(self.vertices, True),
		)


class RegularPolygon(Polygon):
	'''
	Generate an N-sided regular polygon.
	'''

	class Settings(ShapeSettings, total=False):
		''' Settings to be used when generating. '''
		N: int				# number of vertices (randomly generated when N < 3)
		max_vertices: int	# maximum number of vertices when generating

	def __init__(self, N: int = 0, max_vertices: int = 10) -> None:
		super().__init__(_generateRegularPolygon(N if N > 2 else random.randint(3, max_vertices)))


class RegularStar(Polygon):
	'''
	Generate a regular star polygon via a specified Schläfli symbol.
	See: https://en.wikipedia.org/wiki/Star_polygon
	input:
		{p, q} = Schläfli symbol
	'''

	schlafli: tuple[int, int]

	class Settings(ShapeSettings, total=False):
		''' Settings to be used when generating. '''
		schlafli: tuple[int, int]	# Schläfli symbol - (p, q), p >= 3 && p >= (2 * q) + 1
		max_vertices: int			# maximum number of vertices when generating

	def __init__(self, schlafli: tuple[int, int] | None = None, max_vertices: int = 10) -> None:
		p = random.randint(3, max_vertices)
		q = 1 if p > (max_vertices // 2) else random.randint(1, math.floor((p - 1) / 2))
		self.schlafli = (p, q) if schlafli is None else schlafli
		super().__init__(_generateRegularStar(p, q))

	def __getLabels__(self) -> dict[str, list[float | int]]:
		'''
		This method should be used to return the metadata about the current shape.
		'''
		return {'schlafli_symbol': [*self.schlafli], 'N': [self.N()], 'vertices': self.vertices.tolist()}


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
			_normaliseConvexPolygon(self.vertices, True) if self.convex() else _normaliseSimplePolygon(self.vertices, True),
		)


class UnitRectangle(Polygon):
	'''
	Define a rectangle with unit area and an aspect ration epsilon.
	If no default argument is supplied, output is randomly distributed according to ϵ ∈ (0, 1].
	'''

	epsilon: float

	class Settings(ShapeSettings, total=False):
		''' Settings to be used when generating. '''
		epsilon: float		# aspect ratio (randomly generated when epsilon is None)

	def __init__(self, epsilon: float | None = None) -> None:
		self.epsilon = 1. - np.random.uniform(0., 1.) if epsilon is None else epsilon
		super().__init__(_generateUnitRectangle(self.epsilon))

	def __getLabels__(self) -> dict[str, list[float | int]]:
		'''
		This method should be used to return the metadata about the current shape.
		'''
		return {'epsilon': [self.epsilon], 'N': [self.N()], 'vertices': self.vertices.tolist()}


class UnitTriangle(Polygon):
	'''
	Define a triangle with unit area. This construction is achieve through mapping a polar coordinate (r, θ),
	where θ ∈ [0, π / 2] and r ∈ [-1, 1], onto a lens.
	If no default argument is supplied, output is randomly distributed according to r ∈ (0, 1], θ ∈ (0, π).
	'''

	r: float
	theta: float

	class Settings(ShapeSettings, total=False):
		''' Settings to be used when generating. '''
		r: float			# radius (randomly generated when r is None)
		theta: float		# angle (randomly generated when theta is None)

	def __init__(self, r: float | None = None, theta: float | None = None) -> None:
		self.r = 1. - np.random.uniform(0., 1.) if r is None else r
		self.theta = np.random.uniform(np.finfo(float).eps, np.pi) if theta is None else theta
		super().__init__(_generateUnitTriangle(self.r, self.theta))

	def __getLabels__(self) -> dict[str, list[float | int]]:
		'''
		This method should be used to return the metadata about the current shape.
		'''
		return {'r': [self.r], 'N': [self.N()], 'theta': [self.theta], 'vertices': self.vertices.tolist()}
