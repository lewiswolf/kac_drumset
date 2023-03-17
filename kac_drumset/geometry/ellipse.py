'''
This file contains the fixed geometric types used as part of this package.
'''

# dependencies
import cv2					# image processing
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from ..externals._geometry import _circleArea
from .types import Shape, ShapeSettings

__all__ = [
	'Circle',
	'Ellipse',
]


class Ellipse(Shape):
	'''
	A base class for a ellipse, instantiated with two foci.
	'''

	f_1: tuple[float, float]		# focus 1
	f_2: tuple[float, float]		# focus 2

	class Settings(ShapeSettings):
		f_1: tuple[float, float]
		f_2: tuple[float, float]

	def __init__(self, f_1: tuple[float, float], f_2: tuple[float, float]) -> None:
		self.f_1 = f_1
		self.f_2 = f_2

	def area(self) -> float:
		raise Exception('unsupported')
		return 0.

	def centroid(self) -> tuple[float, float]:
		'''
		Return the midpoint between the two foci.
		'''
		return (self.f_1[0] + self.f_2[0]) / 2, (self.f_1[1] + self.f_2[1]) / 2

	def draw(self, grid_size: int) -> npt.NDArray[np.int8]:
		'''
		This function creates a boolean mask of a circle on a grid with dimensions R^(grid_size). The input shape should
		exist within a domain R^G where G ∈ [0, 1].
		'''
		raise Exception('unsupported')
		return cv2.ellipse(
			np.zeros((grid_size, grid_size), 'int8'),
			self.centroid,	#
			(100., 40.),	#
			0.,				#
			0.,
			360.,
			thickness=-1.,
		)

	def isPointInside(self, p: tuple[float, float]) -> bool:
		raise Exception('unsupported')
		return 0.


class Circle(Ellipse):
	'''
	A base class for a circle, instantiated with a radius.
	'''

	_r: float # radius

	class Settings(ShapeSettings):
		r: float

	def __init__(self, r: float = 1.) -> None:
		self.r = r
		super().__init__((0., 0.), (0., 0.))

	def area(self) -> float:
		''' Archimedes. '''
		return _circleArea(self.r)

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
