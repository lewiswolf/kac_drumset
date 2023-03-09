'''
Import functions from external C++ library, housed in geometry/lines.hpp.
'''

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from ..externals._geometry import _lineIntersection

__all__ = [
	'lineIntersection',
]


def lineIntersection(A: npt.NDArray[np.float64], B: npt.NDArray[np.float64]) -> tuple[bool, npt.NDArray[np.float64]]:
	'''
	Finds the point at which two lines intersect.
	collisionLineLine() => https://github.com/bmoren/p5.collide2D
	'''
	assert A.shape == (2, 2) and B.shape == (2, 2), \
		'lineIntersection() only supports an input of A and B with shapes (2, 2).'
	out = _lineIntersection(A, B)
	return out[0], np.array(out[1])
