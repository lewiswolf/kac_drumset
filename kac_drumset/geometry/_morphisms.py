'''
Import functions from external C++ library, housed in geometry/morphisms.hpp.
'''

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from .types import Polygon
from ..externals._geometry import _normaliseConvexPolygon, _normalisePolygon


__all__ = [
	'normaliseConvexPolygon',
	'normalisePolygon',
]


def normaliseConvexPolygon(P: Polygon) -> npt.NDArray[np.float64]:
	'''
	This algorithm produces an identity polygon for each unique polygon given as input. This method normalises an input
	polygon to the unit interval such that x ∈ [0, 1] && y ∈ [0, 1], reducing each input polygon by isometric and
	similarity transformations. This is achieved by first enforcing that the vertices of a polygon are ordered clockwise.
	Then, the largest vector is used to determine the lower and upper bounds across the x-axis. Next, the polygon is split
	into quadrants, the largest of whose area determines the rotation/reflection of the polygon. Finally, the points are
	normalised, and ordered such that V[0] = [0., y].
	'''
	return np.array(_normaliseConvexPolygon(P.vertices))


def normalisePolygon(P: Polygon) -> npt.NDArray[np.float64]:
	'''
	This algorithm performs general normalisation rotations to ensure uniqueness, however it is	not comprehensive for all
	simple geometric transformations.
	'''
	return np.array(_normalisePolygon(P.vertices))
