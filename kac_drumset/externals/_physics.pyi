# core
from typing import Annotated, TypeAlias

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

BooleanImage_1D: TypeAlias = list[int] | npt.NDArray[np.int8]
BooleanImage_2D: TypeAlias = list[list[int]] | npt.NDArray[np.int8]
Matrix_1D: TypeAlias = list[float] | npt.NDArray[np.float64]
Matrix_2D: TypeAlias = list[list[float]] | npt.NDArray[np.float64]
Point: TypeAlias = Annotated[list[float], len(2)] | tuple[float, float]


def _AdditiveSynthesis1D(f: Matrix_1D, alpha: Matrix_1D, d: float, k: float, T: int) -> list[float]: ...
def _AdditiveSynthesis2D(f: Matrix_2D, alpha: Matrix_2D, d: float, k: float, T: int) -> list[float]: ...
def _ChladniPattern1D(U: Matrix_1D, tolerance: float = 0.1) -> list[int]: ...
def _ChladniPattern2D(U: Matrix_2D, tolerance: float = 0.1) -> list[list[int]]: ...
def _circularAmplitudes(r: float, theta: float, S: Matrix_2D) -> list[list[float]]: ...
def _circularCymatics(m: float, n: float, H: int, boundary_conditions: bool = True) -> list[list[float]]: ...
def _circularSeries(M: int, N: int, boundary_conditions: bool = True) -> list[list[float]]: ...
def _equilateralTriangleAmplitudes(x: float, y: float, z: float, N: int, M: int) -> list[list[float]]: ...
def _equilateralTriangleSeries(N: int, M: int) -> list[list[float]]: ...
def _FDTDUpdate1D(
	u_0: Matrix_1D,
	u_1: Matrix_1D,
	c_0: float,
	c_1: float,
	c_2: float,
) -> list[float]: ...
def _FDTDUpdate2D(
	u_0: Matrix_2D,
	u_1: Matrix_2D,
	B: BooleanImage_2D,
	c_0: float,
	c_1: float,
	c_2: float,
	x_range: tuple[int, int],
	y_range: tuple[int, int],
) -> list[list[float]]: ...
def _FDTDWaveform1D(
	u_0: Matrix_1D,
	u_1: Matrix_1D,
	c_0: float,
	c_1: float,
	c_2: float,
	T: int,
	w: float,
) -> list[float]: ...
def _FDTDWaveform2D(
	u_0: Matrix_2D,
	u_1: Matrix_2D,
	B: BooleanImage_2D,
	c_0: float,
	c_1: float,
	c_2: float,
	T: int,
	w: Point,
) -> list[float]: ...
def _raisedCosine1D(mu: float, sigma: float, size: int) -> list[float]: ...
def _raisedCosine2D(mu: Point, sigma: float, size_X: int, size_Y: int) -> list[list[float]]: ...
def _raisedTriangle1D(mu: float, a: float, b: float, size: int) -> list[float]: ...
def _raisedTriangle2D(
	mu: Point,
	x_a: float,
	x_b: float,
	y_a: float,
	y_b: float,
	size_X: int,
	size_Y: int,
) -> list[list[float]]: ...
def _linearAmplitudes(x: float, N: int, boundary_conditions: tuple[bool, bool]) -> list[float]: ...
def _linearCymatics(n: float, X: int, boundary_conditions: tuple[bool, bool]) -> list[float]: ...
def _linearSeries(N: int, boundary_conditions: tuple[bool, bool]) -> list[float]: ...
def _rectangularAmplitudes(
	x: float,
	y: float,
	M: int,
	N: int,
	boundary_conditions: tuple[bool, bool, bool, bool],
) -> list[list[float]]: ...
def _rectangularCymatics(
	m: float,
	n: float,
	X: int,
	Y: int,
	boundary_conditions: tuple[bool, bool, bool, bool],
) -> list[list[float]]: ...
def _rectangularSeries(
	M: int,
	N: int,
	epsilon: float,
	boundary_conditions: tuple[bool, bool, bool, bool],
) -> list[list[float]]: ...
