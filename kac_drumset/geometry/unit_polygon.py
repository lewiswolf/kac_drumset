'''
This file is used to define varying polygons with unit area.
'''

# core
from functools import cached_property

# src
from .types import Polygon

__all__ = [
	'UnitRectangle',
]


class UnitRectangle(Polygon):
	'''
	Define the unit rectangle.
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
