'''
Import functions from external C++ library, housed in geometry/polygon_properties.hpp.
'''

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from .types import Polygon
from ..externals._geometry import (
	_centroid,
	_isColinear,
	_isPointInsideConvexPolygon,
	_largestVector,
)

__all__ = [
	'centroid',
	'isColinear',
	'isPointInsidePolygon',
	'largestVector',
]


def centroid(P: Polygon) -> tuple[float, float]:
	'''
	This algorithm is used to calculate the geometric centroid of a 2D polygon.
	See http://paulbourke.net/geometry/polygonmesh/ 'Calculating the area and centroid of a polygon'.
	'''

	c = _centroid(P.vertices, P.area)
	return c[0], c[1]


def isColinear(vertices: npt.NDArray[np.float64]) -> bool:
	'''
	Function wrapper for _isColinear(), converting (v: npt.NDArray[np.float64]): bool into
	(v: [[float, float], [float, float], [float, float]]): bool.

	Determines whether or not a given set of three vertices are colinear.
	'''
	assert vertices.shape == (3, 2), \
		'isColinear() only supports an input of vertices with shape (3, 2).'
	return _isColinear(vertices)


def isPointInsidePolygon(p: tuple[float, float], P: Polygon) -> bool:
	'''
	Determines whether or not a cartesian point is within a polygon, including boundaries.
	'''
	assert P.convex, 'isPointInsidePolygon() does not currently support concave shapes.'
	return _isPointInsideConvexPolygon(list(p), P.vertices)


def largestVector(P: Polygon) -> tuple[float, tuple[int, int]]:
	'''
	This function tests each pair of vertices in a given polygon to find the largest vector, and returns the length of the
	vector and its indices.
	'''
	out = _largestVector(P.vertices)
	return out[0], (out[1][0], out[1][1])
