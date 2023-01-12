# core
from typing_extensions import TypeAlias

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

Matrix_1D: TypeAlias = list[float] | npt.NDArray[np.float64]
Matrix_2D: TypeAlias = list[list[float]] | npt.NDArray[np.float64]
BooleanImage: TypeAlias = list[list[int]] | npt.NDArray[np.int8]


def _FDTDUpdate2D(
	u_0: Matrix_2D,
	u_1: Matrix_2D,
	B: BooleanImage,
	c_0: float,
	c_1: float,
	c_2: float,
	x_range: tuple[int, int],
	y_range: tuple[int, int],
) -> list[list[float]]: ...


def _FDTDWaveform2D(
	u_0: Matrix_2D,
	u_1: Matrix_2D,
	B: BooleanImage,
	c_0: float,
	c_1: float,
	c_2: float,
	T: int,
	w: tuple[int, int],
) -> list[float]: ...


def _calculateCircularAmplitudes(r: float, theta: float, S: Matrix_2D) -> Matrix_2D: ...
def _calculateCircularSeries(N: int, M: int) -> Matrix_2D: ...
def _calculateRectangularAmplitudes(x: float, y: float, N: int, M: int, epsilon: float) -> Matrix_2D: ...
def _calculateRectangularSeries(N: int, M: int, epsilon: float) -> Matrix_2D: ...
def _raisedCosine1D(size: int, mu: float, sigma: float) -> Matrix_1D: ...
def _raisedCosine2D(size_X: int, size_Y: int, mu_x: float, mu_y: float, sigma: float) -> Matrix_2D: ...
def _raisedTriangle1D(size: int, mu: float, a: float, b: float) -> Matrix_1D: ...
def besselJ(n: float, m: float) -> float: ...
def besselJZero(n: float, m: int) -> float: ...
