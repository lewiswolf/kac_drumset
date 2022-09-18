'''
Import functions from external C++ library and configure python type conversions.
'''

# src
from ..externals._physics import (
	_calculateCircularAmplitudes,
	_calculateCircularSeries,
	_calculateRectangularAmplitudes,
	_calculateRectangularSeries,
	_FDTDWaveform2D,
	_raisedCosine1D,
	_raisedCosine2D,
)

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

__all__ = [
	'calculateCircularAmplitudes',
	'calculateCircularSeries',
	'calculateRectangularAmplitudes',
	'calculateRectangularSeries',
	'FDTDWaveform2D',
	'raisedCosine',
]


def calculateCircularAmplitudes(r: float, theta: float, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
	'''
	Calculate the amplitudes of the circular eigenmodes relative to a polar
	strike location.
	input:
		( r, θ ) = polar strike location
		S = { z_nm | s ∈ ℝ, J_n(z_nm) = 0, 0 <= n < N, 0 < m <= M }
	output:
		A = {
			J_n(z_nm * r) * (2 ** 0.5) * sin(nθπ/4)
			| a ∈ ℝ, J_n(z_nm) = 0, 0 <= n < N, 0 < m <= M
		}
	'''

	return np.array(_calculateCircularAmplitudes(r, theta, S.tolist()))


def calculateCircularSeries(N: int, M: int) -> npt.NDArray[np.float64]:
	'''
	Calculate the eigenmodes of a circle.
	input:
		N = number of modal orders
		M = number of modes per order
	output:
		S = { z_nm | s ∈ ℝ, J_n(z_nm) = 0, n < N, 0 < m <= M }
	'''

	return np.array(_calculateCircularSeries(N, M))


def calculateRectangularAmplitudes(p: tuple[float, float], N: int, M: int, epsilon: float) -> npt.NDArray[np.float64]:
	'''
	Calculate the amplitudes of the rectangular eigenmodes relative to a
	cartesian strike location.
	input:
		( x , y ) = cartesian product
		N = number of modal orders
		M = number of modes per order
		epsilon = aspect ratio of the rectangle
	output:
		A = {
			sin(nyπ / (epsilon ** 0.5)) sin(mxπ / (epsilon ** 0.5))
			| a ∈ ℝ, 0 < n <= N, 0 < m <= M
		}
	'''

	return np.array(_calculateRectangularAmplitudes(
		p[0],
		p[1],
		N,
		M,
		epsilon,
	))


def calculateRectangularSeries(N: int, M: int, epsilon: float) -> npt.NDArray[np.float64]:
	'''
	Calculate the eigenmodes of a rectangle.
	input:
		N = number of modal orders
		M = number of modes per order
		epsilon = aspect ratio of the rectangle
	output:
		S = {
			((m / epsilon)^2 + (n * epsilon)^2) ** 0.5
			| s ∈ ℝ, 0 < n <= N, 0 < m <= M
		}
	'''

	return np.array(_calculateRectangularSeries(N, M, epsilon))


def FDTDWaveform2D(
	u_0: npt.NDArray[np.float64],
	u_1: npt.NDArray[np.float64],
	B: npt.NDArray[np.int8],
	c_0: float,
	c_1: float,
	c_2: float,
	T: int,
	w: tuple[int, int],
) -> npt.NDArray[np.float64]:
	'''
	Generates a waveform using a 2 dimensional FDTD scheme. See `fdtd.hpp` for a parameter description.
	input:
		u_0 = initial fdtd grid at t = 0.
		u_1 = initial fdtd grid at t = 1.
		B = boundary conditions.
		c_0 = first fdtd coefficient related to the decay term and the courant number.
		c_1 = second fdtd coefficient related to the decay term and the courant number.
		c_2 = third fdtd coefficient related to the decay term.
		T = length of simulation in samples.
		w = the coordinate at which the waveform is sampled.
	output:
		waveform = W[n] ∈
			c_0 * (
				u_n_x+1_y + u_n_x-1_y + u_n_x_y+1 + u_n_x_y-1
			) + c_1 * u_n_x_y - c_2 * (u_n-1_x_y) ∀ u ∈ R^2
	'''

	return np.array(_FDTDWaveform2D(
		u_0.tolist(),
		u_1.tolist(),
		B.tolist(),
		c_0,
		c_1,
		c_2,
		T,
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
