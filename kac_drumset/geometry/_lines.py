'''
Import functions from external C++ library, housed in geometry/lines.hpp.
'''

# core
from typing import Literal

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from ..externals._geometry import _isColinear, _largestVector, _lineIntersection, _lineMidpoint

__all__ = [
	'isColinear',
	'largestVector',
	'lineIntersection',
	'lineMidpoint',
]


def isColinear(vertices: npt.NDArray[np.float64]) -> bool:
	'''
	Function wrapper for _isColinear(), converting (v: npt.NDArray[np.float64]): bool into
	(v: [[float, float], [float, float], [float, float]]): bool.

	Determines whether or not a given set of three vertices are colinear.
	'''
	assert vertices.shape == (3, 2), \
		'isColinear() only supports an input of vertices with shape (3, 2).'
	return _isColinear(vertices)


def largestVector(vertices: npt.NDArray[np.float64]) -> tuple[float, tuple[int, int]]:
	'''
	This function tests each pair of vertices in a given polygon to find the largest vector, and returns the length of the
	vector and its indices.
	'''
	assert vertices.ndim == 2 and vertices[0].shape[0] == 2, \
		'largestVector() only supports an input of shape (n, 2).'
	out = _largestVector(vertices)
	return out[0], (out[1][0], out[1][1])


def lineIntersection(A: npt.NDArray[np.float64], B: npt.NDArray[np.float64]) -> tuple[
	Literal['adjacent', 'colinear', 'intersect', 'none', 'vertex'],
	npt.NDArray[np.float64],
]:
	'''
	This function determines whether a line has an intersection, and returns it's type as well
	as the point of intersection (if one exists).
	input
		A, B - Line segments to compare.
	output
		type -
			'none'		No intersection.
			'intersect' The general case where lines intersect one another.
			'vertex'	This is the special case when two lines share a vertex.
			'branch'	This is the special case when a vertex lies within another line. For
						example, B creates an intersection at point B.a when B.a lies on the
						open interval (A.a, A.b).
			'colinear'	This is the special case when the two lines overlap.
		point -
			'none'		Empty point.
			'intersect' The point of intersection ∈ (A.a, A.b) & (B.a, B.b).
			'vertex'	The shared vertex.
			'branch'	The branching vertex.
			'colinear'	The midpoint between all 4 vertices.
	'''
	assert A.shape == (2, 2) and B.shape == (2, 2), \
		'lineIntersection() only supports an input of A and B with shapes (2, 2).'
	out = _lineIntersection(A, B)
	return out[0], np.array(out[1])


def lineMidpoint(A: npt.NDArray[np.float64]) -> tuple[float, float]:
	'''
	Find the midpoint of a line.
	'''
	out = _lineMidpoint(A)
	return out[0], out[1]
