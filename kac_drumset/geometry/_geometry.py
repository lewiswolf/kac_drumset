'''
Import functions from external C++ library and configure python type conversions.
'''

# core
from typing import Optional

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from ..externals._geometry import (
	_centroid,
	_convexNormalisation,
	_generateConvexPolygon,
	_isColinear,
	_isConvex,
	_largestVector,
	_polygonArea,
)

__all__ = [
	'centroid',
	'convexNormalisation',
	'generateConvexPolygon',
	'isColinear',
	'isConvex',
	'largestVector',
	'polygonArea',
]


def centroid(vertices: npt.NDArray[np.float64], area: Optional[float]) -> tuple[float, float]:
	'''
	This algorithm is used to calculate the geometric centroid of a 2D polygon.
	See http://paulbourke.net/geometry/polygonmesh/ 'Calculating the area and
	centroid of a polygon'.
	'''

	if area is None:
		area = polygonArea(vertices)
	c = _centroid(vertices.tolist(), area)
	return (c[0], c[1])


def convexNormalisation(vertices: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
	'''
	This algorithm produces an identity polygon for each unique polygon given as input. This method normalises an input
	polygon to the unit interval such that x ∈ [0, 1] && y ∈ [0, 1], reducing each input polygon by isometric and
	similarity transformations. This is achieved by first enforcing that the vertices of a polygon are ordered clockwise.
	Then, the largest vector is used to determine the lower and upper bounds across the x-axis. Next, the polygon is split
	into quadrants, the largest of whose area determines the rotation/reflection of the polygon. Finally, the points are
	normalised, and ordered such that V[0] = [0., y].
	'''

	return np.array(_convexNormalisation(vertices.tolist()))


def generateConvexPolygon(n: int) -> npt.NDArray[np.float64]:
	'''
	Function wrapper for _genrateConvexPolygon(), converting (n: int): List[[number, number]] into
	(n: int): npt.NDArray[np.float64].

	Generate convex shapes according to Pavel Valtr's 1995 algorithm. Adapted from Sander Verdonschot's Java version,
	found here: https://cglab.ca/~sander/misc/ConvexGeneration/ValtrAlgorithm.java
	'''
	return np.array(_generateConvexPolygon(n))


def isColinear(vertices: npt.NDArray[np.float64]) -> bool:
	'''
	Function wrapper for _isColinear(), converting (v: npt.NDArray[np.float64]): bool into
	(v: [[float, float], [float, float], [float, float]]): bool.

	Determines whether or not a given set of three vertices are colinear.
	'''
	if vertices.shape != (3, 2):
		raise ValueError('isColinear() only supports an input of vertices with shape (3, 2).')
	return _isColinear(vertices.tolist())


def isConvex(vertices: npt.NDArray[np.float64]) -> bool:
	'''
	Function wrapper for _isConvex(), converting (v: npt.NDArray[np.float64]): bool into (v: List[[float, float]]): bool.

	Tests whether or not a given array of vertices forms a convex polygon. This is achieved using the resultant sign of
	the cross product for each vertex: [(x_i - x_i-1), (y_i - y_i-1)] x [(x_i+1 - x_i), (y_i+1 - y_i)].
	See => http://paulbourke.net/geometry/polygonmesh/ 'Determining whether or not a polygon (2D) has its vertices ordered
	clockwise or counter-clockwise'.
	'''
	return _isConvex(vertices.tolist())


def largestVector(vertices: npt.NDArray[np.float64]) -> tuple[float, tuple[int, int]]:
	'''
	This function tests each pair of vertices in a given polygon to find the largest vector, and returns the length of the
	vector and its indices.
	'''
	out = _largestVector(vertices.tolist())
	return out[0], (out[1][0], out[1][1])


def polygonArea(vertices: npt.NDArray[np.float64]) -> float:
	'''
	An implementation of the shoelace algorithm, first described by Albrecht Ludwig Friedrich Meister, which is used to
	calculate the area of a polygon.
	'''
	return _polygonArea(vertices.tolist())
