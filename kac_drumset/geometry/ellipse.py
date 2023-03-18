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
	A base class for an ellipse, instantiated with two foci.
	'''

	f_0: tuple[float, float]
	f_1: tuple[float, float]

	class Settings(ShapeSettings):
		''' Settings to be used when generating. '''
		f_0: tuple[float, float]	# focus §
		f_1: tuple[float, float]	# focus 2

	def __init__(self, f_0: tuple[float, float], f_1: tuple[float, float]) -> None:
		self.f_0 = f_0
		self.f_1 = f_1

	def area(self) -> float:
		'''
		Archimedes.
		'''
		if self.f_0 != self.f_1:
			raise Exception('unsupported')
		return _circleArea(self.r)

	def centroid(self) -> tuple[float, float]:
		'''
		Return the midpoint between the two foci.
		'''
		return (self.f_0[0] + self.f_1[0]) / 2, (self.f_0[1] + self.f_1[1]) / 2

	def draw(self, grid_size: int) -> npt.NDArray[np.int8]:
		'''
		This function creates a boolean mask of a manifold on a grid with dimensions R^(grid_size). The input shape is always
		normalised to the domain R^G before being drawn.
		'''
		if self.f_0 != self.f_1:
			raise Exception('unsupported')
		return cv2.circle(
			np.zeros((grid_size, grid_size), 'int8'),
			(round(grid_size / 2), round(grid_size / 2)),
			round(grid_size / 2),
			1,
			-1,
		)
		# cv2.ellipse(
		# 	np.zeros((grid_size, grid_size), 'int8'),
		# 	self.centroid,	#
		# 	(100., 40.),	#
		# 	0.,				#
		# 	0.,
		# 	360.,
		# 	thickness=-1.,
		# )

	def isPointInside(self, p: tuple[float, float]) -> bool:
		'''
		Determines if a given point p ∈ P, including boundaries.
		'''
		raise Exception('unsupported')
		return 0.


class Circle(Ellipse):
	'''
	A base class for a circle, instantiated with a radius.
	'''

	r: float

	class Settings(ShapeSettings, total=False):
		''' Settings to be used when generating. '''
		r: float			# radius

	def __init__(self, r: float = 1.) -> None:
		self.r = r
		super().__init__((0., 0.), (0., 0.))
