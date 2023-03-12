'''
This file contains functions that project geometric object into alternative representations.
'''

# dependencies
import cv2					# image processing
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from ._polygon_properties import isConvex
from .types import Circle, Polygon

__all__ = [
	'drawCircle',
	'drawPolygon',
]


def drawCircle(C: Circle, grid_size: int) -> npt.NDArray[np.int8]:
	'''
	This function creates a boolean mask of a circle on a grid with dimensions R^(grid_size). The input shape should
	exist within a domain R^G where G ∈ [0, 1].
	'''

	return cv2.circle(
		np.zeros((grid_size, grid_size), 'int8'),
		(round(grid_size / 2), round(grid_size / 2)),
		round(C.r * grid_size / 2),
		1,
		-1,
	)


def drawPolygon(P: Polygon, grid_size: int) -> npt.NDArray[np.int8]:
	'''
	This function creates a boolean mask of a polygon on a grid with dimensions R^(grid_size). The input shape should
	exist within a domain R^G where G ∈ [0, 1].
	'''

	# transposing maintains that mask[x, y] works as intended
	return cv2.fillConvexPoly(
		np.zeros((grid_size, grid_size), 'int8'),
		np.array([[round(y * (grid_size - 1)), round(x * (grid_size - 1))] for [x, y] in P.vertices], 'int32'),
		1,
	) if isConvex(P) else cv2.fillPoly(
		np.zeros((grid_size, grid_size), 'int8'),
		np.array([[[round(y * (grid_size - 1)), round(x * (grid_size - 1))] for [x, y] in P.vertices]], 'int32'),
		1,
	)
