'''
Import modal functions from external C++ library and configure python type conversions.
'''

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from ..externals._physics import (
	_AdditiveSynthesis1D,
	_AdditiveSynthesis2D,
	_ChladniPattern1D,
	_ChladniPattern2D,
	_circularAmplitudes,
	_circularCymatics,
	_circularSeries,
	_equilateralTriangleAmplitudes,
	_equilateralTriangleSeries,
	_linearAmplitudes,
	_linearCymatics,
	_linearSeries,
	_rectangularAmplitudes,
	_rectangularCymatics,
	_rectangularSeries,
)

__all__ = [
	'AdditiveSynthesis',
	'ChladniPattern',
	'circularAmplitudes',
	'circularCymatics',
	'circularSeries',
	'equilateralTriangleAmplitudes',
	'equilateralTriangleSeries',
	'linearAmplitudes',
	'linearCymatics',
	'linearSeries',
	'rectangularAmplitudes',
	'rectangularCymatics',
	'rectangularSeries',
]


def AdditiveSynthesis(
	f: npt.NDArray[np.float64],
	alpha: npt.NDArray[np.float64],
	d: float,
	k: float,
	T: int,
) -> npt.NDArray[np.float64]:
	'''
	Calculate a closed form solution to the 2D wave equation.
	input:
		f = frequencies (hertz)
		α = amplitudes ∈ [-1, 1]
		d = decay
		k = sample length
		T = length of simulation
	output:
		waveform = W[t] ∈ e^dt * sin(ωt) * α
	'''

	assert f.ndim <= 2, \
		'AdditiveSynthesis() only supports one or two dimensional inputs.'
	assert f.ndim == alpha.ndim, \
		'f and alpha must have the same number of dimensions.'
	return np.array(
		_AdditiveSynthesis1D(f, alpha, d, k, T) if f.ndim == 1 else _AdditiveSynthesis2D(f, alpha, d, k, T),
	)


def ChladniPattern(U: npt.NDArray[np.float64], tolerance: float = 0.1) -> npt.NDArray[np.uint8]:
	'''
	Produce a Chladni pattern from a 1 or 2-dimensional cymatic diagram.
	input:
		U = spatial eigenfunction ∈ [-1, 1]
		tolerance = thickness-dependent of the nodal lines
	output:
		B_x = abs(U_x) ≈ 0
		B_xy = abs(U_xy) ≈ 0
	'''

	assert U.ndim <= 2, \
		'ChladniPattern() only supports one or two dimensional inputs.'
	return np.array(
		_ChladniPattern1D(U, tolerance) if U.ndim == 1 else _ChladniPattern2D(U, tolerance),
		dtype=np.uint8,
	)


def circularAmplitudes(r: float, theta: float, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
	'''
	Calculate the spatial eigenfunction of a circular 2-dimensional domain relative to a
	polar strike location. The boundary conditions for this spatial eigenfunction are
	determined by the input series of wavenumbers λ_mn.
	input:
		(r, θ) = polar strike location
		S = { λ_mn | λ ∈ ℝ }
	output:
		α_mn = {
			J_m(λ_mn * r) * √2 * sin(mθ + π/4)
			| α ∈ ℝ, m ∈ [0, M), n ∈ (0, N]
		}
	'''

	return np.array(_circularAmplitudes(r, theta, S))


def circularCymatics(m: float, n: float, H: int, boundary_conditions: bool = True) -> npt.NDArray[np.float64]:
	'''
	Produce the cymatic diagram of a 2-dimensional circular domain for a particular mode λ_mn.
	For creative use, m and n have been defined as real numbers to create continuous animations,
	however for analytics these should be interpreted as integers.
	http://paulbourke.net/geometry/chladni/
	input:
		m = mth modal index
		n = nth modal index
		H = length of the X and Y axes
		boundary_conditions = (true = fixed, false = free)
	output:
		U_rθ = {
			J_n(z_nm * r) * (cos(nθ) + sin(nθ))
			| U ∈ ℝ^2
		}
	'''

	return np.array(_circularCymatics(m, n, H, boundary_conditions))


def circularSeries(M: int, N: int, boundary_conditions: bool = True) -> npt.NDArray[np.float64]:
	'''
	Calculate the wavenumbers of a 2-dimensional circular domain.
	input:
		M = number of modes across the Mth axis
		N = number of modes across the Nth axis
		boundary_conditions = (true = fixed, false = free)
	output:
		λ_mn = {
			J_m(λ_mn) = 0 					dirichlet boundary condition
			J'_m(λ_mn) = 0 					neumann boundary condition
			| λ ∈ ℝ, m ∈ [0, M), n ∈ (0, N]
		}
	'''

	return np.array(_circularSeries(M, N, boundary_conditions))


def equilateralTriangleAmplitudes(u: float, v: float, w: float, N: int, M: int) -> npt.NDArray[np.float64]:
	'''
	Calculate the amplitudes of the equilateral triangle eigenmodes relative to a
	trilinear strike location according to Lamé's formula.
	Seth (1940) Transverse Vibrations of Triangular Membranes.
	input:
		( u, v, w ) = trilinear coordinate
		N = number of modal orders
		M = number of modes per order
	output:
		A = {
			abs(sin(nuπ) sin(nvπ) sin(nwπ))
			| a ∈ ℝ, 0 < n <= N, 0 < m <= M
		}
	'''

	return np.array(_equilateralTriangleAmplitudes(u, v, w, N, M))


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


def linearAmplitudes(
	x: float,
	N: int,
	boundary_conditions: tuple[bool, bool] = (True, True),
) -> npt.NDArray[np.float64]:
	'''
	Calculate the spatial eigenfunction of a 1-dimensional domain relative to a strike location.
	input:
		x = strike location
		N = number of modes
		boundary_conditions = boolean array indicating the boundary conditions
			(true = fixed, false = free)
			1: x-axis minima boundary condition
			2: x-axis maxima boundary condition
	output:
		α_n = {
			sin((n + 1)πx),			dirichlet boundary condition
			cos(nπx),				neumann boundary condition
			sin((n + 0.5)πx),		mixed boundary conditions
			| α ∈ ℝ, n ∈ [0, N)
		}
	'''

	return np.array(_linearAmplitudes(x, N, boundary_conditions))


def linearCymatics(n: float, X: int, boundary_conditions: tuple[bool, bool] = (True, True)) -> npt.NDArray[np.float64]:
	'''
	Produce the cymatic diagram of a 1-dimensional domain for a particular mode λ_n.
	input:
		n = nth modal index
		X = length of the X axis
		boundary_conditions = boolean array indicating the boundary conditions
			(true = fixed, false = free)
			1: x-axis minima boundary condition
			2: x-axis maxima boundary condition
	output:
		U_x = {
			sin((n + 1) πx/H),		dirichlet boundary condition
			cos(nπx/H),				neumann boundary condition
			sin((n + 0.5)πx/H),		mixed boundary conditions
			| U ∈ ℝ^1, n ∈ [0, ∞)
		}
	'''

	return np.array(_linearCymatics(n, X, boundary_conditions))


def linearSeries(N: int, boundary_conditions: tuple[bool, bool] = (True, True)) -> npt.NDArray[np.float64]:
	'''
	Calculate the wavenumbers of a 1-dimensional domain.
	input:
		N = number of modes
		boundary_conditions = boolean array indicating the boundary conditions
			(true = fixed, false = free)
			1: x-axis minima boundary condition
			2: x-axis maxima boundary condition
	output:
		λ_n = {
			n + 1,					dirichlet boundary condition
			n,						neumann boundary condition
			n + 0.5,				mixed boundary conditions
			| λ ∈ ℝ, n ∈ [0, N)
		}
	'''

	return np.array(_linearSeries(N, boundary_conditions))


def rectangularAmplitudes(
	p: tuple[float, float],
	M: int,
	N: int,
	epsilon: float,
	boundary_conditions: tuple[bool, bool, bool, bool] = (True, True, True, True),
) -> npt.NDArray[np.float64]:
	'''
	Calculate the spatial eigenfunction of a rectangular 2-dimensional domain relative to a
	cartesian strike location.
	input:
		(x, y) = cartesian strike location
		M = number of modes across the Mth axis
		N = number of modes across the Nth axis
		epsilon = aspect ratio of the rectangle
		boundary_conditions = boolean array indicating the boundary conditions
			(true = fixed, false = free)
			1: x-axis minima boundary condition
			2: x-axis maxima boundary condition
			3: y-axis minima boundary condition
			4: y-axis maxima boundary condition
	output:
		X_m = {
			sin((m + 1)xπ / √Є),	dirichlet boundary condition
			cos(mxπ / √Є),			neumann boundary condition
			sin((m + 0.5)xπ / √Є),	mixed boundary conditions
			| m ∈ [0, M)
		}
		Y_n = {
			sin((n + 1)yπ√Є),		dirichlet boundary condition
			cos(nyπ√Є),				neumann boundary condition
			sin((n + 0.5)yπ√Є),		mixed boundary conditions
			| n ∈ [0, N)
		}
		α_mn = { X_m * Y_n | α ∈ ℝ }
	'''

	return np.array(_rectangularAmplitudes(p[0], p[1], M, N, epsilon, boundary_conditions))


def rectangularCymatics(
	m: float,
	n: float,
	X: int,
	Y: int,
	boundary_conditions: tuple[bool, bool, bool, bool] = (True, True, True, True),
) -> npt.NDArray[np.float64]:
	'''
	Produce the cymatic diagram of a 2-dimensional rectangular domain for a particular
	mode λ_mn.
	For creative use, m and n have been defined as real numbers to create continuous animations,
	however for analytics these should be interpreted as integers.
	input:
		m = mth modal index
		n = nth modal index
		X = length of the X axis
		Y = length of the Y axis
		boundary_conditions = boolean array indicating the boundary conditions
			(true = fixed, false = free)
			1: x-axis minima boundary condition
			2: x-axis maxima boundary condition
			3: y-axis minima boundary condition
			4: y-axis maxima boundary condition
	output:
		X_m = {
			sin((m + 1)xπ / X),		dirichlet boundary condition
			cos(mxπ / X),			neumann boundary condition
			sin((m + 0.5)xπ / X),	mixed boundary conditions
			| m ∈ [0, ∞)
		}
		Y_n = {
			sin((n + 1)yπ / Y),		dirichlet boundary condition
			cos(nyπ / Y),			neumann boundary condition
			sin((n + 0.5)yπ / Y),	mixed boundary conditions
			| n ∈ [0, ∞)
		}
		U_xy = { X_m * Y_n | U ∈ ℝ^2 }
	'''

	return np.array(_rectangularCymatics(m, n, X, Y, boundary_conditions))


def rectangularSeries(
	M: int,
	N: int,
	epsilon: float,
	boundary_conditions: tuple[bool, bool, bool, bool] = (True, True, True, True),
) -> npt.NDArray[np.float64]:
	'''
	Calculate the wavenumbers of a 2-dimensional rectangular domain.
	input:
		M = number of modes across the Mth axis
		N = number of modes across the Nth axis
		epsilon = aspect ratio of the rectangle
		boundary_conditions = boolean array indicating the boundary conditions
			(true = fixed, false = free)
			1: x-axis minima boundary condition
			2: x-axis maxima boundary condition
			3: y-axis minima boundary condition
			4: y-axis maxima boundary condition
	output:
		X_m = {
			(m + 1)^2 / Є,			dirichlet boundary condition
			m^2 / Є,				neumann boundary condition
			(m + 0.5)^2 / Є,		mixed boundary conditions
			| m ∈ [0, M)
		}
		Y_n = {
			(n + 1)^2 * Є,			dirichlet boundary condition
			n^2 * Є,				neumann boundary condition
			(n + 0.5)^2 * Є,		mixed boundary conditions
			| n ∈ [0, N)
		}
		λ_mn = { √(X_m + Y_n) | λ ∈ ℝ }
	'''

	return np.array(_rectangularSeries(N, M, epsilon, boundary_conditions))
