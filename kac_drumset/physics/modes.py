'''
Import modal functions from external C++ library and configure python type conversions.
'''

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from ..externals._physics import (
	_calculateCircularAmplitudes,
	_calculateCircularSeries,
	_calculateRectangularAmplitudes,
	_calculateRectangularSeries,
	_WaveEquationWaveform2D,
)

__all__ = [
	'calculateCircularAmplitudes',
	'calculateCircularSeries',
	'calculateRectangularAmplitudes',
	'calculateRectangularSeries',
	'WaveEquationWaveform2D',
]


def calculateCircularAmplitudes(r: float, theta: float, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
	'''
	Calculate the amplitudes of the circular eigenmodes relative to a polar strike location.
	input:
		( r, θ ) = polar strike location
		S = { z_nm | s ∈ ℝ, J_n(z_nm) = 0, 0 <= n < N, 0 < m <= M }
	output:
		A = {
			J_n(z_nm * r) * (2 ** 0.5) * sin(nθπ/4)
			| a ∈ ℝ, J_n(z_nm) = 0, 0 <= n < N, 0 < m <= M
		}
	'''

	return np.array(_calculateCircularAmplitudes(r, theta, S))


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
	Calculate the amplitudes of the rectangular eigenmodes relative to a cartesian strike location.
	input:
		( x , y ) = cartesian product
		N = number of modal orders
		M = number of modes per order
		epsilon = aspect ratio of the rectangle
	output:
		A = {
			sin(mxπ / (Є ** 0.5)) sin(nyπ * (Є ** 0.5))
			| a ∈ ℝ, 0 < n <= N, 0 < m <= M
		}
	'''

	return np.array(_calculateRectangularAmplitudes(p[0], p[1], N, M, epsilon))


def calculateRectangularSeries(N: int, M: int, epsilon: float) -> npt.NDArray[np.float64]:
	'''
	Calculate the eigenmodes of a rectangle.
	input:
		N = number of modal orders
		M = number of modes per order
		epsilon = aspect ratio of the rectangle
	output:
		S = {
			((m ** 2 / Є) + (Єn ** 2)) ** 0.5
			| s ∈ ℝ, 0 < n <= N, 0 < m <= M
		}
	'''

	return np.array(_calculateRectangularSeries(N, M, epsilon))


def WaveEquationWaveform2D(
	F: npt.NDArray[np.float64],
	A: npt.NDArray[np.float64],
	d: float,
	k: float,
	T: int,
) -> npt.NDArray[np.float64]:
	'''
	Calculate a closed form solution to the 2D wave equation.
	input:
		F = frequencies (hertz)
		A = amplitudes ∈ [0, 1]
		d = decay
		k = sample length
		T = length of simulation
	output:
		waveform = W[n] ∈ A * e^dt * sin(Ft) / N * max(A)
	'''

	return np.array(_WaveEquationWaveform2D(F, A, d, k, T))
