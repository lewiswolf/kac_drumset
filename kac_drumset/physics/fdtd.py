'''
Import FDTD functions from external C++ library and configure python type conversions.
'''

# core
from typing import Optional, cast

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from ..externals._physics import (
	_FDTDUpdate1D,
	_FDTDUpdate2D,
	_FDTDWaveform1D,
	_FDTDWaveform2D,
	_raisedCosine1D,
	_raisedCosine2D,
	_raisedTriangle1D,
	_raisedTriangle2D,
)

__all__ = [
	# methods
	'FDTDWaveform1D',
	'FDTDWaveform2D',
	'raisedCosine',
	'raisedTriangle',
	# classes
	'FDTD',
]


class FDTD():
	'''
	Class implementation of a two dimensional FDTD equation. This method is designed to be used as an iterator:
	for u in FDTD(*args):
		print(u)
	input:
		u_0 = initial fdtd grid at t = 0.
		u_1 = initial fdtd grid at t = 1.
		c_0 = first fdtd coefficient related to the decay term and the courant number.
		c_1 = second fdtd coefficient related to the decay term and the courant number.
		c_2 = third fdtd coefficient related to the decay term.
		T = length of simulation.
		B = boundary condition (Optional: only incorporated when u_0 = 2D).
	output:
	output:
		u_0 = 1D:
			u[n] = u = c_0 * (u_n-1_x+1 + u_n-1_x-1) + c_1 * u_n-1_x - c_2 * u_n-2_x
		u_0 = 2D:
			u[n] = c_0 * (u_x+1_y + u_0_x-1_y + u_0_x_y+1 + u_0_x_y-1) + c_1 * u_0_x_y - c_2 * (u_1_x_y)
	'''

	_n: int
	c_0: float
	c_1: float
	c_2: float
	T: int
	u_0: list[float] | list[list[float]]
	u_1: list[float] | list[list[float]]
	B: list[int] | list[list[int]]
	x_range: tuple[int, int]
	y_range: tuple[int, int]

	def __init__(
		self,
		u_0: npt.NDArray[np.float64],
		u_1: npt.NDArray[np.float64],
		c_0: float,
		c_1: float,
		c_2: float,
		T: int,
		B: Optional[npt.NDArray[np.int8]] = None,
	) -> None:
		''' Initialise FDTD iterator. '''

		# initialise domains
		assert u_0.ndim <= 2, \
			'FDTD() only supports one or two dimensional inputs.'
		self.u_0 = u_0.tolist()
		self.u_1 = u_1.tolist()
		# decay coefficients
		self.c_0 = c_0
		self.c_1 = c_1
		self.c_2 = c_2
		# define simulation length
		self.T = T
		# boundary conditions
		self.B = []
		self.x_range = (0, 0)
		self.y_range = (0, 0)
		if u_0.ndim == 2:
			X, Y = u_0.shape
			if B is None:
				self.B = np.pad(np.ones((X - 2, Y - 2), dtype=np.int8), 1, mode='constant').tolist()
			else:
				self.B = B.tolist()
				x_range = [X, 0]
				y_range = [Y, 0]
				for x in range(0, X):
					for y in range(0, Y):
						# forward loop to find the first ones
						if (B[x][y] == 1):
							x_range[0] = x if x_range[0] > x else x_range[0]
							y_range[0] = y if y_range[0] > y else y_range[0]
							continue
				for x in range(X - 2, 0, -1):
					for y in range(Y - 2, 0, -1):
						# backwards loop to find the last ones
						if (B[x][y] == 1):
							x_range[1] = x if x_range[1] < x else x_range[1]
							y_range[1] = y if y_range[1] < y else y_range[1]
							continue
			self.x_range = (x_range[0], x_range[1])
			self.y_range = (y_range[0], y_range[1])

	def __iter__(self) -> 'FDTD':
		''' Return the iterator. '''
		self._n = 0
		return self

	def __next__(self) -> npt.NDArray[np.float64]:
		''' Compute the FDTD update equation at every iteration. '''

		if self._n < self.T:
			self._n += 1
			self.u_0 = _FDTDUpdate1D(
				cast(list[float], self.u_0),
				cast(list[float], self.u_1),
				self.c_0,
				self.c_1,
				self.c_2,
			) if isinstance(self.u_0[0], float) else _FDTDUpdate2D(
				cast(list[list[float]], self.u_0),
				cast(list[list[float]], self.u_1),
				cast(list[list[int]], self.B),
				self.c_0,
				self.c_1,
				self.c_2,
				self.x_range,
				self.y_range,
			)
			self.u_0, self.u_1 = self.u_1, self.u_0
			return np.asarray(self.u_1)
		else:
			raise StopIteration


def FDTDWaveform1D(
	u_0: npt.NDArray[np.float64],
	u_1: npt.NDArray[np.float64],
	c_0: float,
	c_1: float,
	c_2: float,
	T: int,
	w: float,
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
		w = the coordinate at which the waveform is sampled ∈ ℝ^2, [0. 1.].
	output:
		waveform = W[n] ∈
			c_0 * (u_n_x+1_y + u_n_x-1_y + u_n_x_y+1 + u_n_x_y-1) + c_1 * u_n_x_y - c_2 * (u_n-1_x_y) ∀ u ∈ R^2
	'''

	return np.array(_FDTDWaveform1D(u_0, u_1, c_0, c_1, c_2, T, w))


def FDTDWaveform2D(
	u_0: npt.NDArray[np.float64],
	u_1: npt.NDArray[np.float64],
	B: npt.NDArray[np.int8],
	c_0: float,
	c_1: float,
	c_2: float,
	T: int,
	w: tuple[float, float],
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
		w = the coordinate at which the waveform is sampled ∈ ℝ^2, [0. 1.].
	output:
		waveform = W[n] ∈
			c_0 * (u_n_x+1_y + u_n_x-1_y + u_n_x_y+1 + u_n_x_y-1)
				+ c_1 * u_n_x_y - c_2 * (u_n-1_x_y) ∀ u ∈ R^2
	'''

	return np.array(_FDTDWaveform2D(u_0, u_1, B, c_0, c_1, c_2, T, w))


def raisedCosine(
	mu: tuple[float] | tuple[float, float],
	matrix_size: tuple[int, ...],
	sigma: float = 0.25,
) -> npt.NDArray[np.float64]:
	'''
	Calculate a one or two-dimensional raised cosine distribution, normalised to a unit interval.
	input:
		μ = a normalised point representing the maxima of the distribution ∈ [0, 1].
		matrix_size = A tuple representing the size of the output matrix.
		σ = normalised half-width of the distribution ∈ (0, ∞].
	output:
		RC(x):
			{
				(1 + cos(π * |x - μ| / σ)) / 2,	|x - μ| ≤ σ
				0,								|x - μ| > σ
			}
		RC(x, y):
			l2_norm = ((x - mu_x)^2 + (y - mu_y)^2)^0.5
			{
				(1 + cos(π * l2_norm / σ)) / 2,	l2_norm ≤ σ
				0,								l2_norm > σ
			}
	'''

	assert len(mu) <= 2 and len(mu) == len(matrix_size), \
		'raisedCosine() only supports one or two dimensional inputs.'
	return np.array(_raisedCosine1D(
		mu[0],
		sigma,
		matrix_size[0],
	) if len(mu) == 1 else np.array(_raisedCosine2D(
		mu,
		sigma,
		matrix_size[0],
		matrix_size[1],
	)))


def raisedTriangle(
	mu: tuple[float] | tuple[float, float],
	matrix_size: tuple[int, ...],
	x_ab: tuple[float, float] = (0.25, 0.25),
	y_ab: tuple[float, float] = (0.25, 0.25),
) -> npt.NDArray[np.float64]:
	'''
	Calculate a one or two-dimensional triangular function, normalised to a unit interval.
	input:
		μ = a normalised point representing the maxima of the distribution ∈ [0, 1].
		size = the size of the matrix.
		x_a = normalised segment length of horizontal distribution such that a = μ - x_a.
		x_b = normalised segment length of horizontal distribution such that b = μ - x_b.
		y_a = normalised segment length of vertical distribution such that a = μ - y_a.
		y_b = normalised segment length of vertical distribution such that b = μ - y_b.
	output:
		Λ(x, y) = Λ(x) * Λ(y)
		Λ(x) = {
			0,								x < a
			(x - a) / (μ - a),				a ≤ x ≤ μ
			1. - (x - μ) / (b - μ),			μ < x ≤ b
			0,								x > b
		}
	'''

	assert len(mu) <= 2 and len(mu) == len(matrix_size), \
		'raisedTriangle() only supports one or two dimensional inputs.'
	return np.array(_raisedTriangle1D(
		mu[0],
		x_ab[0],
		x_ab[1],
		matrix_size[0],
	)) if len(mu) == 1 else np.array(_raisedTriangle2D(
		mu,
		x_ab[0],
		x_ab[1],
		y_ab[0],
		y_ab[1],
		matrix_size[0],
		matrix_size[1],
	))
