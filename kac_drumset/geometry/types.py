'''
This file contains the fixed geometric types used as part of this package.
'''

# core
from abc import ABC, abstractmethod

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

__all__ = [
	'Shape',
]


class Shape(ABC):
	'''
	An abstract base class for a two dimensional manifold in Euclidean geometry.
	'''

	def __init__(self) -> None:
		pass

	@abstractmethod
	def area(self) -> float:
		pass

	@abstractmethod
	def centroid(self) -> tuple[float, float]:
		pass

	@abstractmethod
	def draw(self, grid_size: int) -> npt.NDArray[np.int8]:
		pass

	@abstractmethod
	def isPointInside(self, p: tuple[float, float]) -> bool:
		pass
