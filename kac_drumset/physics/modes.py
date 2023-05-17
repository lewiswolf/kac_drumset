'''
Import modal functions from external C++ library and configure python type conversions.
'''

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from ..externals._physics import (
	_circularAmplitudes,
	_circularSeries,
	_equilateralTriangleAmplitudes,
	_equilateralTriangleSeries,
	_rectangularAmplitudes,
	_rectangularChladniPattern,
	_rectangularSeries,
	_WaveEquationWaveform2D,
)

__all__ = [
	'circularAmplitudes',
	'circularSeries',
	'equilateralTriangleAmplitudes',
	'equilateralTriangleSeries',
	'rectangularAmplitudes',
	'rectangularChladniPattern',
	'rectangularSeries',
	'WaveEquationWaveform2D',
]


def circularAmplitudes(r: float, theta: float, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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

	return np.array(_circularAmplitudes(r, theta, S))


def circularSeries(N: int, M: int) -> npt.NDArray[np.float64]:
	'''
	Calculate the eigenmodes of a circle.
	input:
		N = number of modal orders
		M = number of modes per order
	output:
		S = { z_nm | s ∈ ℝ, J_n(z_nm) = 0, n < N, 0 < m <= M }
	'''

	return np.array(_circularSeries(N, M))


def equilateralTriangleAmplitudes(x: float, y: float, z: float, N: int, M: int) -> npt.NDArray[np.float64]:
	'''
	Calculate the amplitudes of the equilateral triangle eigenmodes relative to a
	trilinear strike location according to Lamé's formula.
	Seth (1940) Transverse Vibrations of Triangular Membranes.
	input:
		( x, y, z ) = trilinear coordinate
		N = number of modal orders
		M = number of modes per order
	output:
		A = {
			abs(sin(nxπ) sin(nyπ) sin(nzπ))
			| a ∈ ℝ, 0 < n <= N, 0 < m <= M
		}
	'''

	return np.array(_equilateralTriangleAmplitudes(x, y, z, N, M))


def equilateralTriangleSeries(N: int, M: int) -> npt.NDArray[np.float64]:
	'''
	Calculate the eigenmodes of an equilateral triangle according to Lamé's formula.
	Seth (1940) Transverse Vibrations of Triangular Membranes.
	input:
		N = number of modal orders
		M = number of modes per order
	output:
		S = {
			(m ** 2 + n ** 2 + mn) ** 0.5
			| s ∈ ℝ, 0 < n <= N, 0 < m <= M
		}
	'''

	return np.array(_equilateralTriangleSeries(N, M))


def rectangularAmplitudes(p: tuple[float, float], N: int, M: int, epsilon: float) -> npt.NDArray[np.float64]:
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

	return np.array(_rectangularAmplitudes(p[0], p[1], N, M, epsilon))


def rectangularChladniPattern(m: int, n: int, X: int, Y: int, tolerance: float = 0.1) -> npt.NDArray[np.float64]:
	'''
	Produce the 2D chladni pattern for a rectangular plate.
	http://paulbourke.net/geometry/chladni/
	input:
		m = mth modal index
		n = nth modal index
		X = length of the X axis
		Y = length of the Y axis
		tolerance = the standard deviation between the calculation and the final pattern
	output:
		M = {
			cos(nπx/X) cos(mπy/Y) - cos(mπx/X) cos(nπy/Y)
		}
	'''

	return np.array(_rectangularChladniPattern(m, n, X, Y, tolerance))


def rectangularSeries(N: int, M: int, epsilon: float) -> npt.NDArray[np.float64]:
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

	return np.array(_rectangularSeries(N, M, epsilon))


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
		waveform = W[n] ∈ A * e^dt * sin(Ft) / NM * max(A)
	'''

	return np.array(_WaveEquationWaveform2D(F, A, d, k, T))
