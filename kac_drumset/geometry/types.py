'''
This file contains the abstract base class for every two-dimensional shape.
'''

# core
from abc import ABC, abstractmethod
from typing import TypedDict

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

__all__ = [
	'Shape',
	'ShapeSettings',
]


class ShapeSettings(TypedDict, total=False):
	''' Placeholder for custom ShapeSettings. '''
	pass


class Shape(ABC):
	'''
	An abstract base class for a two dimensional manifold in Euclidean geometry.
	'''

	def __init__(self) -> None:
		pass

	@abstractmethod
	class Settings(ShapeSettings, total=False):
		'''
		Settings to be used when generating.
		'''
		pass

	@abstractmethod
	def __getLabels__(self) -> dict[str, list[float | int]]:
		'''
		This method should be used to return the metadata about the current shape.
		'''
		pass

	@property
	@abstractmethod
	def area(self) -> float:
		'''
		Calculate the area of a 2D manifold. This property should be used to scale the shape whenever it is set.
		'''
		pass

	@property
	@abstractmethod
	def centroid(self) -> tuple[float, float]:
		'''
		This algorithm is used to calculate the geometric centroid of a 2D manifold. This property should be used move the
		shape about the plane whenever it is set.
		'''
		pass

	@abstractmethod
	def draw(self, grid_size: int) -> npt.NDArray[np.int8]:
		'''
		This function creates a boolean mask of a manifold on a grid with dimensions R^(grid_size). The input shape is always
		normalised to the domain R^G before being drawn.
		'''
		pass

	@abstractmethod
	def isPointInside(self, p: tuple[float, float]) -> bool:
		'''
		Determines if a given point p ∈ P, including boundaries.
		'''
		pass
