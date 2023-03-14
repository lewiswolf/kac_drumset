'''
This file is used to define varying polygons with unit area.
'''

# src
# from ..externals._geometry import _generateUnitRectangle, _generateUnitTriangle
from ..externals._geometry import _generateUnitRectangle
from .types import Polygon

__all__ = [
	'UnitRectangle',
	'UnitTriangle',
]


class UnitRectangle(Polygon):
	'''
	Define a rectangle with unit area and an aspect ration epsilon.
	'''

	epsilon: float

	def __init__(self, epsilon: float = 1.) -> None:
		self.epsilon = epsilon
		super().__init__(_generateUnitRectangle(epsilon))

	def area(self) -> float: return 1.


class UnitTriangle(Polygon):
	'''
	Define a triangle with unit area. For any point (r, θ) where θ ∈ [0, π / 2] and r ∈ [0, 1], the corresponding triangle
	will be unique.
	'''

	def __init__(self, r: float, theta: float) -> None:
		self.r = r
		assert self.r <= 1. and self.r >= 0., 'r ∈ [0, 1]'
		self.theta = theta
		# super().__init__(_generateUnitTriangle(r, theta))

	def area(self) -> float: return 1.

# r_sin_theta = self.r * math.sin(self.theta)
# super().__init__(np.array([
# 	[1 / r_sin_theta, -1 * r_sin_theta],
# 	[-1 / r_sin_theta, -1 * r_sin_theta],
# 	[
# 		self.r * math.cos((2 * self.theta / 3) - np.pi / 3) - 0.5,
# 		self.r * math.sin((2 * self.theta / 3) - np.pi / 3) - 0.5,
# 	] if math.floor(self.theta / np.pi) % 2 == 0 else [
# 		self.r * math.cos((2 * self.theta / 3) + np.pi / 3) - 0.5,
# 		self.r * math.sin((2 * self.theta / 3) + np.pi / 3) - 0.5,
# 	],
# 	# [one_root_two * self.r * math.cos(self.theta), one_root_two * r_sin_theta],
# 	# [one_root_two * self.r * math.cos(self.theta), one_root_two * self.r * math.sin(self.theta - (np.pi / 6))],
# ]) / (2 ** 0.5))
