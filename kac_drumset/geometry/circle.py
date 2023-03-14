'''
This file contains the fixed geometric types used as part of this package.
'''

# dependencies
import cv2					# image processing
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from .types import Shape

__all__ = [
	'Circle',
]


class Circle(Shape):
	'''
	A base class for a circle, instantiated with a radius.
	'''

	r: float # radius

	def __init__(self, r: float = 1.) -> None:
		self.r = r

	def area(self) -> float:
		''' Archimedes. '''
		return np.pi * (self.r ** 2.)

	def centroid(self) -> tuple[float, float]:
		return 0., 0.

	def draw(self, grid_size: int) -> npt.NDArray[np.int8]:
		'''
		This function creates a boolean mask of a circle on a grid with dimensions R^(grid_size). The input shape should
		exist within a domain R^G where G ∈ [0, 1].
		'''
		return cv2.circle(
			np.zeros((grid_size, grid_size), 'int8'),
			(round(grid_size / 2), round(grid_size / 2)),
			round(self.r * grid_size / 2),
			1,
			-1,
		)

	def isPointInside(self, p: tuple[float, float]) -> bool:
		return 1. >= p[0] and p[1] >= -1.
