'''
RandomPolygon is used to generate a random drum shape alongside computing some of its properties.
'''

# core
import random
from typing import Optional

# dependencies
import numpy as np 			# maths

# src
from ..externals._geometry import (
	_generateConvexPolygon,
	_generateUnitRectangle,
	_generateIrregularStar,
	_generatePolygon,
	_normaliseConvexPolygon,
	_normalisePolygon,
)
from .polygon import Polygon
from .types import ShapeSettings

__all__ = [
	'ConvexPolygon',
	'IrregularStar',
	'TSPolygon',
	'UnitRectangle',
	'UnitTriangle',
]


class ConvexPolygon(Polygon):
	'''
	'''

	class Settings(ShapeSettings, total=False):
		N: int
		max_vertices: int

	def __init__(self, N: Optional[int] = None, max_vertices: int = 10) -> None:
		super().__init__(_normaliseConvexPolygon(_generateConvexPolygon(
			random.randint(3, max_vertices) if N is None else N,
		)))


class IrregularStar(Polygon):
	'''
	'''

	class Settings(ShapeSettings, total=False):
		N: int
		max_vertices: int

	def __init__(self, N: Optional[int] = None, max_vertices: int = 10) -> None:
		super().__init__(_generateIrregularStar(
			random.randint(3, max_vertices) if N is None else N,
		))
		self.vertices = np.array(_normaliseConvexPolygon(self.vertices) if self.convex else _normalisePolygon(self.vertices))


class TSPolygon(Polygon):
	'''
	'''

	class Settings(ShapeSettings, total=False):
		N: int
		max_vertices: int

	def __init__(self, N: Optional[int] = None, max_vertices: int = 10) -> None:
		super().__init__(_generatePolygon(
			random.randint(3, max_vertices) if N is None else N,
		))
		self.vertices = np.array(_normaliseConvexPolygon(self.vertices) if self.convex else _normalisePolygon(self.vertices))


class UnitRectangle(Polygon):
	'''
	Define a rectangle with unit area and an aspect ration epsilon.
	'''

	epsilon: float

	class Settings(ShapeSettings, total=False):
		epsilon: float

	def __init__(self, epsilon: Optional[float] = None) -> None:
		self.epsilon = np.random.uniform(0., 1.) if epsilon is None else epsilon
		super().__init__(_generateUnitRectangle(self.epsilon))


class UnitTriangle(Polygon):
	'''
	Define a triangle with unit area. For any point (r, θ) where θ ∈ [0, π / 2] and r ∈ [0, 1], the corresponding triangle
	will be unique.
	'''

	r: float
	theta: float

	class Settings(ShapeSettings, total=False):
		r: float
		theta: float

	def __init__(self, r: float, theta: float) -> None:
		self.r = np.random.uniform(0., 1.) if r is None else r
		self.theta = np.random.uniform(0., np.pi) if theta is None else theta
		assert self.r <= 1. and self.r >= 0., 'r ∈ [0, 1]'
		# super().__init__(_generateUnitTriangle(r, theta))
