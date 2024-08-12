'''
This file contains the fixed geometric types used as part of this package.
'''

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
	major: float					# length across the x axis
	minor: float					# length across the y axis

	class Settings(ShapeSettings, total=False):
		''' Settings to be used when generating. '''
		major: float				# length across the x axis
		minor: float				# length across the y axis (randomly generated when minor = 0.)

	def __init__(self, major: float = 1., minor: float = 0., centroid: tuple[float, float] = (0., 0.)) -> None:
		minor = minor or np.random.uniform(0., 1.)
		if (major >= minor):
			self.major = major
			self.minor = minor
		else:
			self.major = minor
			self.minor = major
		self._centroid = centroid

	'''
	Getters and setters for area. Setting area scales the ellipse.
	'''
	@property
	def area(self) -> float:
		''' Archimedes '''
		return self.major * self.minor * np.pi

	@area.setter
	def area(self, a: float) -> None:
		epsilon = self.major / self.minor
		scaled_a = ((a / np.pi) ** 0.5)
		self.major = scaled_a * epsilon
		self.minor = scaled_a / epsilon

	'''
	Getters and setters for centroid. Setting centroid translates the ellipse about the plane.
	'''
	@property
	def centroid(self) -> tuple[float, float]:
		return self._centroid

	@centroid.setter
	def centroid(self, value: tuple[float, float]) -> None:
		self._centroid = value

	def __getLabels__(self) -> dict[str, list[float | int]]:
		'''
		This method should be used to return the metadata about the current shape.
		'''
		return {'major': [self.major], 'minor': [self.minor]}

	def draw(self, grid_size: int) -> npt.NDArray[np.int8]:
		'''
		This function creates a boolean mask of a manifold on a grid with dimensions R^(grid_size). The input shape is always
		normalised to the domain R^G before being drawn.
		'''
		half_grid = round((grid_size - 1) / 2)
		return cv2.circle(
			np.zeros((grid_size, grid_size), np.int8),
			(half_grid, half_grid),
			half_grid,
			(1, 0, 0),
			-1,
		).astype(np.int8) if self.major == self.minor else cv2.ellipse(
			np.zeros((grid_size, grid_size), np.int8),
			(half_grid, half_grid),
			# transposing maintains that mask[x, y] works as intended
			(round(half_grid * self.minor / self.major), half_grid),
			0,
			0,
			360,
			(1, 0, 0),
			thickness=-1,
		).astype(np.int8)

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
		c = self.focalDistance()
		return ((self.centroid[0], self.centroid[1] + c), (self.centroid[0], self.centroid[1] - c))

	def focalDistance(self) -> float:
		'''
		The distance between a focus and the centroid.
		'''
		return self.major * self.eccentricity()

	def isPointInside(self, p: tuple[float, float]) -> bool:
		'''
		Determines if a given point p ∈ P, including boundaries.
		'''
		major_2 = (self.major ** 2)
		minor_2 = (self.minor ** 2)
		return (((p[0] - self.centroid[0]) ** 2) * minor_2) \
			+ (((p[1] - self.centroid[1]) ** 2) * major_2) <= (major_2 * minor_2)


class Circle(Ellipse):
	'''
	A base class for a circle, instantiated with a radius.
	'''

	class Settings(ShapeSettings, total=False):
		''' Settings to be used when generating. '''
		r: float			# radius (randomly generated when r = 0)

	def __init__(self, r: float = 0., centroid: tuple[float, float] = (0., 0.)) -> None:
		r = r or np.random.uniform(0., 1.)
		super().__init__(r, r, centroid)

	'''
	Getters and setters for radius. Updating the radius updates both major and minor.
	'''
	@property
	def r(self) -> float:
		assert self.major == self.minor
		return self.major

	@r.setter
	def r(self, value: float) -> None:
		self.major = value
		self.minor = value

	def __getLabels__(self) -> dict[str, list[float | int]]:
		'''
		This method should be used to return the metadata about the current shape.
		'''
		return {'r': [self.r], 'major': [self.major], 'minor': [self.minor]}
