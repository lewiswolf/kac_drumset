# core
from typing_extensions import TypeAlias

Matrix_1D: TypeAlias = list[float]
Matrix_2D: TypeAlias = list[list[float]]


def _FDTDUpdate2D(
	u_0: Matrix_2D,
	u_1: Matrix_2D,
	B: list[list[int]],
	c_0: float,
	c_1: float,
	c_2: float,
	x_range: tuple[int, int],
	y_range: tuple[int, int],
) -> Matrix_2D: ...


def _FDTDWaveform2D(
	u_0: Matrix_2D,
	u_1: Matrix_2D,
	B: list[list[int]],
	c_0: float,
	c_1: float,
	c_2: float,
	T: int,
	w: tuple[int, int],
) -> Matrix_1D: ...


def _calculateCircularAmplitudes(r: float, theta: float, S: Matrix_2D) -> Matrix_2D: ...
def _calculateCircularSeries(N: int, M: int) -> Matrix_2D: ...
def _calculateRectangularAmplitudes(x: float, y: float, N: int, M: int, epsilon: float) -> Matrix_2D: ...
def _calculateRectangularSeries(N: int, M: int, epsilon: float) -> Matrix_2D: ...
def _raisedCosine1D(size: int, mu: float, sigma: float) -> Matrix_1D: ...
def _raisedCosine2D(size_X: int, size_Y: int, mu_x: float, mu_y: float, sigma: float) -> Matrix_2D: ...
def _raisedTriangle1D(size: int, mu: float, a: float, b: float) -> Matrix_1D: ...
def besselJ(n: float, m: float) -> float: ...
def besselJZero(n: float, m: int) -> float: ...
