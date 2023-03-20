'''
This file contains the fixed geometric types used as part of this package.
'''

# core
from typing import Optional

# dependencies
import cv2					# image processing
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from .types import Shape, ShapeSettings

__all__ = [
	'Circle',
	'Ellipse',
]


class Ellipse(Shape):
	'''
	A base class for an ellipse, instantiated with two foci.
	'''

	_centroid: tuple[float, float]	# center of the ellipse
	major: float
	minor: float

	class Settings(ShapeSettings):
		''' Settings to be used when generating. '''
		major: tuple[float, float]	# length across the x axis
		minor: tuple[float, float]	# length across the y axis

	def __init__(self, major: float, minor: float, centroid: tuple[float, float] = (0., 0.)) -> None:
		self.major = major if major > minor else minor
		self.minor = minor if major > minor else major
		self._centroid = centroid

	'''
	'''

	@property
	def centroid(self) -> tuple[float, float]:
		return self._centroid

	@centroid.setter
	def centroid(self, value: tuple[float, float]) -> None:
		self._centroid = value

	def area(self) -> float:
		''' Archimedes '''
		return self.major * self.minor * np.pi

	def draw(self, grid_size: int) -> npt.NDArray[np.int8]:
		'''
		This function creates a boolean mask of a manifold on a grid with dimensions R^(grid_size). The input shape is always
		normalised to the domain R^G before being drawn.
		'''
		half_grid = round((grid_size - 1) / 2)
		return cv2.circle(
			np.zeros((grid_size, grid_size), 'int8'),
			(half_grid, half_grid),
			half_grid,
			1,
			-1,
		) if self.major == self.minor else cv2.ellipse(
			np.zeros((grid_size, grid_size), 'int8'),
			(half_grid, half_grid),
			(half_grid, round(half_grid * self.minor / self.major)),
			0,
			0,
			360,
			1,
			thickness=-1,
		)

	def eccentricity(self) -> float:
		'''
		The ratio between the focal distance and the major axis.
		'''
		return (1. - (self.minor ** 2. / self.major ** 2.)) ** 0.5

	def foci(self) -> tuple[tuple[float, float], tuple[float, float]]:
		'''
		The foci are the two points at which the sum of the distances between any point on the surface of the ellipse is a
		constant.
		'''
		c = self.focal_distance()
		return ((self.centroid[0], self.centroid[1] + c), (self.centroid[0], self.centroid[1] - c))

	def focal_distance(self) -> float:
		'''
		The distance between a focus and the origin.
		'''
		return self.major * self.eccentricity()

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

	_r: float

	class Settings(ShapeSettings, total=False):
		''' Settings to be used when generating. '''
		r: float			# radius

	def __init__(self, r: Optional[float] = None, centroid: tuple[float, float] = (0., 0.)) -> None:
		self._r = np.random.uniform(0., 1.) if r is None else r
		super().__init__(self._r, self._r, centroid)

	@property
	def r(self) -> float:
		assert self.major == self._r and self.minor == self._r
		return self._r

	@r.setter
	def r(self, value: float) -> None:
		self._r = value
		self.major = value
		self.minor = value
