'''
Import functions from external C++ library and configure python type conversions.
'''

# src
from ..externals._physics import _raisedCosine1D, _raisedCosine2D

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy


def raisedCosine(
	mu: tuple[int, ...],
	matrix_size: tuple[int, ...],
	sigma: float = 0.5,
) -> npt.NDArray[np.float64]:
	'''
	This functions creates a raised cosine distribution, centred at the mu. Only 1D and 2D distributions are supported.
	params:
			mu				The coordinate used to represent the centre of the
							cosine distribution.
			matrix_size		A tuple representing the size of the output matrix.
			sigma			The radius of the distribution.
	'''

	if len(mu) > 2 or len(mu) != len(matrix_size):
		# handle dimensions > 2 and incompatible inputs
		raise ValueError('raisedCosine() only supports one or two dimensional inputs.')
	if len(mu) == 1:
		return np.array(_raisedCosine1D(mu[0], matrix_size[0], sigma))
	else:
		return np.array(_raisedCosine2D(
			mu[0],
			mu[1],
			matrix_size[0],
			matrix_size[1],
			sigma,
		))
