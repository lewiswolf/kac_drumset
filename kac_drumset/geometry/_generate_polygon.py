'''
Import functions from external C++ library, housed in geometry/generate_polygon.hpp.
'''

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from ..externals._geometry import _generateIrregularStar, _generateConvexPolygon, _generatePolygon

__all__ = [
	'generateIrregularStar',
	'generateConvexPolygon',
	'generatePolygon',
]


def generateIrregularStar(N: int) -> npt.NDArray[np.float64]:
	'''
	This is a fast method for generating concave polygons, particularly with a large number of vertices. This approach
	generates polygons by ordering a series of random points around a centre point. As a result, not all possible simple
	polygons are generated this way.
	input:
		N = the number of vertices
		seed? = the seed for the random number generators
	output:
		P = an irregular star of N random vertices
	'''
	return np.array(_generateIrregularStar(N))


def generateConvexPolygon(N: int) -> npt.NDArray[np.float64]:
	'''
	Function wrapper for _genrateConvexPolygon(), converting (N: int): List[[number, number]] into
	(N: int): npt.NDArray[np.float64].

	Generate convex shapes according to Pavel Valtr's 1995 algorithm. Adapted from Sander Verdonschot's Java version,
	found here: https://cglab.ca/~sander/misc/ConvexGeneration/ValtrAlgorithm.java
	'''
	return np.array(_generateConvexPolygon(N))


def generatePolygon(N: int) -> npt.NDArray[np.float64]:
	'''
	This algorithm is based on a method of eliminating self-intersections in a polygon by using the Lin and Kerningham
	'2-opt' moves. Such a move eliminates an intersection between two edges by reversing the order of the vertices between
	the edges. Intersecting edges are detected using a simple sweep through the vertices and then one intersection is
	chosen at random to eliminate after each sweep.
	https://doc.cgal.org/latest/Generator/group__PkgGeneratorsRef.html#gaa8cb58e4cc9ab9e225808799b1a61174
	van Leeuwen, J., & Schoone, A. A. (1982). Untangling a traveling salesman tour in the plane.
	input:
		N = the number of vertices
		seed? = the seed for the random number generators
	output:
		P = a concave polygon of N random vertices
	'''
	return np.array(_generatePolygon(N))
