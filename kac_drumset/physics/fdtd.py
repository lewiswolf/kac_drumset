'''
Import FDTD functions from external C++ library and configure python type conversions.
'''

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from ..externals._physics import (
	_FDTDUpdate2D,
	_FDTDWaveform2D,
	_raisedCosine1D,
	_raisedCosine2D,
	_raisedTriangle1D,
	_raisedTriangle2D,
)

__all__ = [
	# methods
	'FDTDWaveform2D',
	'raisedCosine',
	'raisedTriangle',
	# classes
	'FDTD_2D',
]


class FDTD_2D():
	'''
	Class implementation of a two dimensional FDTD equation. This method is designed to be used as an iterator:
	for u in FDTD(*args):
		print(u)
	input:
		u_0 = initial fdtd grid at t = 0.
		u_1 = initial fdtd grid at t = 1.
		B = boundary condition.
		c_0 = first fdtd coefficient related to the decay term and the courant number.
		c_1 = second fdtd coefficient related to the decay term and the courant number.
		c_2 = third fdtd coefficient related to the decay term.
		T = length of simulation.
	output:
		u[n] = c_0 * (
			u_x+1_y + u_0_x-1_y + u_0_x_y+1 + u_0_x_y-1
		) + c_1 * u_0_x_y - c_2 * (u_1_x_y)
	'''

	_n: int
	B: list[list[int]]
	c_0: float
	c_1: float
	c_2: float
	T: int
	u_0: list[list[float]]
	u_1: list[list[float]]
	x_range: tuple[int, int]
	y_range: tuple[int, int]

	def __init__(
		self,
		u_0: list[list[float]],
		u_1: list[list[float]],
		B: list[list[int]],
		c_0: float,
		c_1: float,
		c_2: float,
		T: int,
	) -> None:
		''' Initialise FDTD iterator. '''

		# initialise domains
		self.u_0 = u_0
		self.u_1 = u_1
		self.B = B
		# decay coefficients
		self.c_0 = c_0
		self.c_1 = c_1
		self.c_2 = c_2
		# define simulation length
		self.T = T
		# calculate x_range and y_range
		x_range = [len(B), 0]
		y_range = [len(B[0]), 0]
		for x in range(0, len(B)):
			for y in range(0, len(B[0])):
				# forward loop to find the first ones
				if (B[x][y] == 1):
					x_range[0] = x if x_range[0] > x else x_range[0]
					y_range[0] = y if y_range[0] > y else y_range[0]
					continue
		for x in range(len(B) - 2, 0, -1):
			for y in range(len(B[0]) - 2, 0, -1):
				# backwards loop to find the last ones
				if (B[x][y] == 1):
					x_range[1] = x if x_range[1] < x else x_range[1]
					y_range[1] = y if y_range[1] < y else y_range[1]
					continue
		self.x_range = (x_range[0], x_range[1])
		self.y_range = (y_range[0], y_range[1])

	def __iter__(self) -> 'FDTD_2D':
		''' Return the iterator. '''
		self._n = 0
		return self

	def __next__(self) -> npt.NDArray[np.float64]:
		''' Compute the FDTD update equation at every iteration. '''

		if self._n < self.T:
			self._n += 1
			if self._n % 2 == 1:
				self.u_0 = _FDTDUpdate2D(
					self.u_0,
					self.u_1,
					self.B,
					self.c_0,
					self.c_1,
					self.c_2,
					self.x_range,
					self.y_range,
				)
				return np.asarray(self.u_0)
			else:
				self.u_1 = _FDTDUpdate2D(
					self.u_1,
					self.u_0,
					self.B,
					self.c_0,
					self.c_1,
					self.c_2,
					self.x_range,
					self.y_range,
				)
				return np.asarray(self.u_1)
		else:
			raise StopIteration


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
			c_0 * (
				u_n_x+1_y + u_n_x-1_y + u_n_x_y+1 + u_n_x_y-1
			) + c_1 * u_n_x_y - c_2 * (u_n-1_x_y) ∀ u ∈ R^2
	'''

	return np.array(_FDTDWaveform2D(u_0, u_1, B, c_0, c_1, c_2, T, w))


def raisedCosine(
	matrix_size: tuple[int, ...],
	mu: tuple[float] | tuple[float, float],
	sigma: float = 0.5,
) -> npt.NDArray[np.float64]:
	'''
	Creates a raised cosine distribution centred at mu. Only 1D and 2D distributions are supported.
	input:
		matrix_size = A tuple representing the size of the output matrix.
		μ = The coordinate used to represent the centre of the cosine distribution.
		σ = The radius of the distribution.
	'''

	assert len(mu) <= 2 and len(mu) == len(matrix_size), \
		'raisedCosine() only supports one or two dimensional inputs.'
	return np.array(_raisedCosine1D(
		matrix_size[0],
		mu[0],
		sigma,
	) if len(mu) == 1 else _raisedCosine2D(
		matrix_size[0],
		matrix_size[1],
		mu,
		sigma,
	))


def raisedTriangle(
	matrix_size: tuple[int, ...],
	mu: tuple[float] | tuple[float, float],
	x_ab: tuple[float, float] | None = None,
	y_ab: tuple[float, float] | None = None,
) -> npt.NDArray[np.float64]:
	'''
	Calculate a one or two dimensional triangular distribution.
	input:
		size = the size of the matrix.
		μ = a cartesian point representing the maxima of the triangle.
		x_ab = minimum and maximum x value for the distribution.
		y_ab = minimum and maximum y value for the distribution.
	output:
		Λ(x, y) = Λ(x) * Λ(y)
		Λ(x) = {
			0,								x < a
			(x - a) / (μ - a),				a ≤ x ≤ μ
			1. - (x - μ) / (b - μ),			μ < x ≤ b
			0,								x > a
		}
	'''

	assert len(mu) <= 2 and len(mu) == len(matrix_size), \
		'raisedTriangle() only supports one or two dimensional inputs.'
	# configure x_ab
	if x_ab is None:
		x_ab = (0, matrix_size[0] - 1)
	assert x_ab[0] <= mu[0] and x_ab[1] >= mu[0]
	if len(mu) == 1:
		return np.array(_raisedTriangle1D(
			matrix_size[0],
			mu[0],
			x_ab[0],
			x_ab[1],
		))
	else:
		# configure y_ab
		if y_ab is None:
			y_ab = (0, matrix_size[1] - 1)
		assert y_ab[0] <= mu[1] and y_ab[1] >= mu[1]
		return np.array(_raisedTriangle2D(
			matrix_size[0],
			matrix_size[1],
			mu,
			x_ab[0],
			x_ab[1],
			y_ab[0],
			y_ab[1],
		))
