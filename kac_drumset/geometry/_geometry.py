# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from ..externals._geometry import _generateConvexPolygon, _isConvex


def generateConvexPolygon(n: int) -> npt.NDArray[np.float64]:
	'''
	Function wrapper for _genrateConvexPolygon(), converting (n: int): List[[number, number]] into
	(n: int): npt.NDArray[np.float64].
	'''
	return np.array(_generateConvexPolygon(n))


def isConvex(vertices: npt.NDArray[np.float64]) -> bool:
	'''
	Function wrapper for _genrateConvexPolygon(), converting (v: npt.NDArray[np.float64]): bool into
	(v: List[[number, number]]): bool.
	'''
	return _isConvex(vertices.tolist())
