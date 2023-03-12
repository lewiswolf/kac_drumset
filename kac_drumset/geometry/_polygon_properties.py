'''
Import functions from external C++ library, housed in geometry/polygon_properties.hpp.
'''

# src
from .types import Polygon
from ..externals._geometry import (
	_centroid,
	_isConvex,
	_isPointInsideConvexPolygon,
	_isSimple,
	_largestVector,
)

__all__ = [
	'centroid',
	'isConvex',
	'isPointInsidePolygon',
	'isSimple',
	'largestVector',
]


def centroid(P: Polygon) -> tuple[float, float]:
	'''
	This algorithm is used to calculate the geometric centroid of a 2D polygon.
	See http://paulbourke.net/geometry/polygonmesh/ 'Calculating the area and centroid of a polygon'.
	'''

	c = _centroid(P.vertices, P.area())
	return c[0], c[1]


def isConvex(P: Polygon) -> bool:
	'''
	Tests whether or not a given polygon is convex. This is achieved using the resultant sign of the cross product for
	each vertex:
		[(x_i - x_i-1), (y_i - y_i-1)] × [(x_i+1 - x_i), (y_i+1 - y_i)].
	See => http://paulbourke.net/geometry/polygonmesh/ 'Determining whether or not a polygon (2D) has its vertices ordered
	clockwise or counter-clockwise'.
	'''
	return _isConvex(P.vertices)


def isPointInsidePolygon(p: tuple[float, float], P: Polygon) -> bool:
	'''
	Determines whether or not a cartesian point is within a polygon, including boundaries.
	'''
	assert isConvex(P), 'isPointInsidePolygon() does not currently support concave shapes.'
	return _isPointInsideConvexPolygon(list(p), P.vertices)


def isSimple(P: Polygon) -> bool:
	'''
	Determine if a polygon is simple by checking for intersections.
	'''
	return _isSimple(P.vertices)


def largestVector(P: Polygon) -> tuple[float, tuple[int, int]]:
	'''
	This function tests each pair of vertices in a given polygon to find the largest vector, and returns the length of the
	vector and its indices.
	'''
	out = _largestVector(P.vertices)
	return out[0], (out[1][0], out[1][1])
