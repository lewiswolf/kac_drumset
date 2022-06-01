'''
Import functions from external C++ library and configure python type conversions.
'''

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from ..externals._geometry import _generateConvexPolygon, _isColinear, _isConvex

__all__ = [
	'generateConvexPolygon',
	'isColinear',
	'isConvex',
]


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
