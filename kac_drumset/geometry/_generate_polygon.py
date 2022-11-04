'''
Import functions from external C++ library, housed in geometry/generate_polygon.hpp.
'''

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from ..externals._geometry import _generateConvexPolygon

__all__ = ['generateConvexPolygon']


def generateConvexPolygon(N: int) -> npt.NDArray[np.float64]:
	'''
	Function wrapper for _genrateConvexPolygon(), converting (n: int): List[[number, number]] into
	(n: int): npt.NDArray[np.float64].

	Generate convex shapes according to Pavel Valtr's 1995 algorithm. Adapted from Sander Verdonschot's Java version,
	found here: https://cglab.ca/~sander/misc/ConvexGeneration/ValtrAlgorithm.java
	'''
	return np.array(_generateConvexPolygon(N))
