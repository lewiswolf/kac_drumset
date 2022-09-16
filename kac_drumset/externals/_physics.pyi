# core
from typing_extensions import TypeAlias

Matrix_1D: TypeAlias = list[float]
Matrix_2D: TypeAlias = list[list[float]]


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
def _calculateCircularSeries(N: int, M: int) -> Matrix_2D: ...
def _raisedCosine1D(size: int, mu: int, sigma: float) -> Matrix_1D: ...
def _raisedCosine2D(size_X: int, size_Y: int, mu_x: int, mu_y: int, sigma: float) -> Matrix_2D: ...
def besselJ(n: float, m: float) -> float: ...
def besselJZero(n: float, m: int) -> float: ...
