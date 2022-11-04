'''
This file contains functions that project geometric object into alternative representations.
'''

# core
from typing import Optional

# dependencies
import cv2					# image processing
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from ._polygon_properties import isConvex
from .types import Polygon

__all__ = ['booleanMask']


def booleanMask(P: Polygon, grid_size: int, convex: Optional[bool] = None) -> npt.NDArray[np.int8]:
	'''
	This function creates a boolean mask of a polygon on a grid with dimensions R^(grid_size). The input shape should
	exist within a domain R^G where G ∈ [0, 1].
	'''

	if convex is None:
		convex = isConvex(P)

	mask = np.zeros((grid_size, grid_size), 'int8')
	if convex:
		cv2.fillConvexPoly(
			mask,
			np.array([[
				round(x * (grid_size - 1)),
				round(y * (grid_size - 1)),
			] for [x, y] in P.vertices], 'int32'),
			1,
		)
	else:
		cv2.fillPoly(
			mask,
			np.array([[[
				round(x * (grid_size - 1)),
				round(y * (grid_size - 1)),
			] for [x, y] in P.vertices]], 'int32'),
			1,
		)
	return np.transpose(mask) # this maintains that mask[x, y] works as intended
