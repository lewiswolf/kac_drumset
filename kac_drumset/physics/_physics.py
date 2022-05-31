'''
Import functions from external C++ library and configure python type conversions.
'''

# src
from ..externals._physics import _FDTDWaveform2D, _raisedCosine1D, _raisedCosine2D

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy


def FDTDWaveform2D(
	u_0: npt.NDArray[np.float64],
	u_1: npt.NDArray[np.float64],
	B: npt.NDArray[np.int8],
	c_0: float,
	c_1: float,
	d: float,
	T: int,
	x_range: tuple[int, int],
	y_range: tuple[int, int],
	w: tuple[float, float],
) -> npt.NDArray[np.float64]:
	'''
	Generates a waveform using a 2 dimensional FDTD scheme. See `fdtd.hpp` for a parameter description.
	'''

	return np.array(_FDTDWaveform2D(
		u_0.tolist(),
		u_1.tolist(),
		B.tolist(),
		c_0,
		c_1,
		d,
		T,
		x_range,
		y_range,
		w,
	))


def raisedCosine(
	matrix_size: tuple[int, ...],
	mu: tuple[int, ...],
	sigma: float = 0.5,
) -> npt.NDArray[np.float64]:
	'''
	This function creates a raised cosine distribution centred at mu. Only 1D and 2D distributions are supported.
	input:
		matrix_size		A tuple representing the size of the output matrix.
		mu				The coordinate used to represent the centre of the
						cosine distribution.
		sigma			The radius of the distribution.
	'''

	if len(mu) > 2 or len(mu) != len(matrix_size):
		# handle dimensions > 2 and incompatible inputs
		raise ValueError('raisedCosine() only supports one or two dimensional inputs.')
	if len(mu) == 1:
		return np.array(_raisedCosine1D(matrix_size[0], mu[0], sigma))
	else:
		return np.array(_raisedCosine2D(
			matrix_size[0],
			matrix_size[1],
			mu[0],
			mu[1],
			sigma,
		))
