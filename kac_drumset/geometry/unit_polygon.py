'''
This file is used to define varying polygons with unit area.
'''

# core
from functools import cached_property
import math

# src
from .types import Polygon

__all__ = [
	'UnitRectangle',
	'UnitTriangle',
]


class UnitRectangle(Polygon):
	'''
	Define a rectangle with unit area.
	'''

	def __init__(self, epsilon: float = 1.) -> None:
		'''
		input:
			epsilon = aspect ratio
		'''

		self.epsilon = epsilon
		x_0 = 0.5 * self.epsilon
		y_0 = 0.5 / self.epsilon
		x_1 = -1. * x_0
		y_1 = -1. * y_0
		super().__init__([[x_0, y_0], [x_0, y_1], [x_1, y_1], [x_1, y_0]])

	@cached_property
	def area(self) -> float: return 1.


class UnitTriangle(Polygon):
	'''
	Define a triangle with unit area.
	'''

	def __init__(self, r: float, theta: float) -> None:
		'''
		For any point (r, θ) where θ ∈ [0, π / 2] and r ∈ [0, 1], the corresponding triangle will be unique.
		'''

		self.r = r
		assert self.r <= 1. and self.r >= 0., 'r ∈ [0, 1]'
		self.theta = theta
		r_sin_theta = self.r * math.sin(self.theta)
		one_root_two = 1 / (2 ** 0.5)
		super().__init__([
			[one_root_two / r_sin_theta, -one_root_two * r_sin_theta],
			[-one_root_two / r_sin_theta, -one_root_two * r_sin_theta],
			[one_root_two * self.r * math.cos(self.theta), one_root_two * r_sin_theta],
		])

	@cached_property
	def area(self) -> float: return 1.
