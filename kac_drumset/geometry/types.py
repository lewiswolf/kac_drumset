'''
This file contains the fixed geometric types used as part of this package.
'''

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

__all__ = [
	'Polygon',
]


class Polygon():
	'''
	A base class for a polygon, instantiated with an array of vertices.
	'''

	n: int								# number of vertices
	vertices: npt.NDArray[np.float64]	# cartesian products representing the corners of a shape

	def __init__(self, vertices: npt.NDArray[np.float64]) -> None:
		if (vertices.ndim != 2 or vertices.shape[1] != 2):
			raise ValueError('Array of vertices is not the correct shape: (n, 2)')
		self.vertices = vertices
		self.n = vertices.shape[0]
