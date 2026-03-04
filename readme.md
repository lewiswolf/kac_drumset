<!-- ![Kac-Drumming](https://user-images.githubusercontent.com/55607290/169860844-7f3f3d6d-4366-4410-8a30-5ee9472c2864.png) -->

# kac_drumset

![python version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
<a href="https://doi.org/10.5281/zenodo.7274474">
![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.7274474-blue)
</a>

Python based analysis tools and dataset generator for arbitrarily shaped drums.

# Install

```bash
pip install "git+https://github.com/lewiswolf/kac_drumset.git"
```

### Dependencies

-	[cmake](https://formulae.brew.sh/formula/cmake)
-   [libsndfile](https://github.com/libsndfile/libsndfile)

# Core Library

<details>
<summary>Geometry</summary>

### Import

```python
from kac_drumset.geometry import (
	# Methods
	isColinear,
	largestVector,
	lineIntersection,
	weylCondition,
	# Classes
	Circle,
	ConvexPolygon,
	IrregularStar,
	RegularPolygon,
	TravellingSalesmanPolygon,
	UnitRectangle,
	UnitTriangle,
	# Types
	Ellipse,
	Polygon,
	Shape,
	ShapeSettings,
)
```

### Methods

```python
def isColinear(vertices: npt.NDArray[np.float64]) -> bool:
	'''
	Determines whether or not a given set of three vertices are colinear.
	'''

def largestVector(vertices: npt.NDArray[np.float64]) -> tuple[float, tuple[int, int]]:
	'''
	This function tests each pair of vertices in a given set of points to find the largest vector, and returns the length
	of the vector and its indices.
	'''

def lineIntersection(A: npt.NDArray[np.float64], B: npt.NDArray[np.float64]) -> tuple[
	Literal['branch', 'colinear', 'intersect', 'none', 'vertex'],
	npt.NDArray[np.float64],
]:
	'''
	This function determines whether a line has an intersection, and returns it's type as well
	as the point of intersection (if one exists).
	input:
		A, B - Line segments to compare.
	output:
		type -
			'none'		No intersection.
			'intersect' The general case where lines intersect one another.
			'vertex'	This is the special case when two lines share a vertex.
			'branch'	This is the special case when a vertex lies within another line. For
						example, B creates an intersection at point B.a when B.a lies on the
						open interval (A.a, A.b).
			'colinear'	This is the special case when the two lines overlap.
		point -
			'none'		Empty point.
			'intersect' The point of intersection ∈ (A.a, A.b) & (B.a, B.b).
			'vertex'	The shared vertex.
			'branch'	The branching vertex.
			'colinear'	The midpoint between all 4 vertices.
	'''

def weylCondition(S_1: Shape, S_2: Shape) -> bool:
	'''
	Using Weyl's asymptotic law, determine whether two polygons may be isospectral.
	https://en.wikipedia.org/wiki/Weyl_law
	'''
```

### Classes

```python

class Circle(Ellipse):
	'''
	A base class for a circle, instantiated with a radius.
	'''

	class Settings(ShapeSettings, total=False):
		''' Settings to be used when generating. '''
		r: float			# radius (randomly generated when r = 0)

	def __init__(self, r: float = 0., centroid: tuple[float, float] = (0., 0.)) -> None:

class ConvexPolygon(Polygon):
	'''
	Adapted from Sander Verdonschot's Java version, found here:
	https://cglab.ca/~sander/misc/ConvexGeneration/ValtrAlgorithm.java
	'''

	class Settings(ShapeSettings, total=False):
		''' Settings to be used when generating. '''
		N: int				# number of vertices (randomly generated when N < 3)
		max_vertices: int	# maximum number of vertices when generating

	def __init__(self, N: int = 0, max_vertices: int = 10) -> None:

class IrregularStar(Polygon):
	'''
	This is a fast method for generating concave polygons, particularly with a large number of vertices. This approach
	generates polygons by ordering a series of random points around a centre point. As a result, not all possible simple
	polygons are generated this way.
	'''

	class Settings(ShapeSettings, total=False):
		''' Settings to be used when generating. '''
		N: int				# number of vertices (randomly generated when N < 3)
		max_vertices: int	# maximum number of vertices when generating

	def __init__(self, N: int = 0, max_vertices: int = 10) -> None:

class RegularPolygon(Polygon):
	'''
	Generate an N-sided regular polygon.
	'''

	class Settings(ShapeSettings, total=False):
		''' Settings to be used when generating. '''
		N: int				# number of vertices (randomly generated when N < 3)
		max_vertices: int	# maximum number of vertices when generating

	def __init__(self, N: int = 0, max_vertices: int = 10) -> None:

class TravellingSalesmanPolygon(Polygon):
	'''
	This algorithm is based on a method of eliminating self-intersections in a polygon by using the Lin and Kerningham
	'2-opt' moves. Such a move eliminates an intersection between two edges by reversing the order of the vertices between
	the edges. Intersecting edges are detected using a simple sweep through the vertices and then one intersection is
	chosen at random to eliminate after each sweep.
	van Leeuwen, J., & Schoone, A. A. (1982). Untangling a traveling salesman tour in the plane.
	'''

	class Settings(ShapeSettings, total=False):
		''' Settings to be used when generating. '''
		N: int				# number of vertices (randomly generated when N < 3)
		max_vertices: int	# maximum number of vertices when generating

	def __init__(self, N: int = 0, max_vertices: int = 10) -> None:

class UnitRectangle(Polygon):
	'''
	Define a rectangle with unit area and an aspect ration epsilon.
	If no default argument is supplied, output is randomly distributed according to ϵ ∈ (0, 1].
	'''

	class Settings(ShapeSettings, total=False):
		''' Settings to be used when generating. '''
		epsilon: float		# aspect ratio (randomly generated when epsilon is None)

	def __init__(self, epsilon: float | None = None) -> None:


class UnitTriangle(Polygon):
	'''
	Define a triangle with unit area. This construction is achieve through mapping a polar coordinate (r, θ),
	where θ ∈ [0, π / 2] and r ∈ [-1, 1], onto a lens.
	If no default argument is supplied, output is randomly distributed according to r ∈ (0, 1], θ ∈ (0, π).
	'''

	class Settings(ShapeSettings, total=False):
		''' Settings to be used when generating. '''
		r: float			# radius (randomly generated when r is None)
		theta: float		# angle (randomly generated when theta is None)

	def __init__(self, r: float | None = None, theta: float | None = None) -> None:
```

### Types

```python
class Ellipse(Shape):
	'''
	A base class for an ellipse, instantiated with two foci.
	'''

	major: float			# length across the x axis
	minor: float			# length across the y axis

	class Settings(ShapeSettings, total=False):
		''' Settings to be used when generating. '''
		major: float		# length across the x axis
		minor: float		# length across the y axis (randomly generated when minor = 0.)

	def __init__(self, major: float = 1., minor: float = 0., centroid: tuple[float, float] = (0., 0.)) -> None:

	@property
	def area(self) -> float:
		'''
		Getters and setters for area. Setting area scales the ellipse.
		'''

	@property
	def centroid(self) -> tuple[float, float]:
		'''
		Getters and setters for centroid. Setting centroid translates the ellipse about the plane.
		'''

	def draw(self, grid_size: int) -> npt.NDArray[np.int8]:
		'''
		This function creates a boolean mask of a manifold on a grid with dimensions R^(grid_size). The input shape is always
		normalised to the domain R^G before being drawn.
		'''

	def eccentricity(self) -> float:
		'''
		The ratio between the focal distance and the major axis.
		'''

	def foci(self) -> tuple[tuple[float, float], tuple[float, float]]:
		'''
		The foci are the two points at which the sum of the distances between any point on the surface of the ellipse is a
		constant.
		'''

	def focalDistance(self) -> float:
		'''
		The distance between a focus and the centroid.
		'''

	def isPointInside(self, p: tuple[float, float]) -> bool:
		'''
		Determines if a given point p ∈ P, including boundaries.
		'''

class Polygon(Shape):
	'''
	A base class for a polygon, instantiated with an array of vertices.
	'''

	class Settings(ShapeSettings, total=False):
		''' Settings to be used when generating. '''
		vertices: list[list[float]] | npt.NDArray[np.float64]

	def __init__(self, vertices: list[list[float]] | npt.NDArray[np.float64]) -> None:

	@property
	def area(self) -> float:
		'''
		An implementation of the polygon area algorithm derived using Green's Theorem.
		Setting the area scales the polygon, whilst preserving its centroid.
		'''

	@property
	def centroid(self) -> tuple[float, float]:
		'''
		Getters and setters for centroid. Setting centroid translates the polygon about the plane.
		'''

	@property
	def vertices(self) -> npt.NDArray[np.float64]:
		'''
		The vertices of the polygon, here exposed as a mutable property.
		'''

	def convex(self) -> bool:
		'''
		Determine whether or not the polygon is convex. The convexity of the polygon is cached when the vertices are set.
		This is to save time when computing other Class methods such as draw() and isPointInside().
		'''

	def draw(self, grid_size: int) -> npt.NDArray[np.int8]:
		'''
		This function creates a boolean mask of a manifold on a grid with dimensions R^(grid_size). The input shape is always
		normalised to the domain R^G before being drawn.
		'''

	def N(self) -> int:
		'''
		Return the number of vertices for the polygon.
		'''

	def isPointInside(self, p: tuple[float, float]) -> bool:
		'''
		Determines if a given point p ∈ P, including boundaries.
		'''

	def simple(self) -> bool:
		'''
		Determine whether or not the polygon is simple by checking for intersections.
		'''

class Shape(ABC):
	'''
	An abstract base class for a two dimensional manifold in Euclidean geometry.
	'''

	def __init__(self) -> None:
		pass

	@abstractmethod
	class Settings(ShapeSettings, total=False):
		'''
		Settings to be used when generating.
		'''

	@abstractmethod
	def __getLabels__(self) -> dict[str, list[float | int]]:
		'''
		This method should be used to return the metadata about the current shape.
		'''

	@property
	@abstractmethod
	def area(self) -> float:
		'''
		Calculate the area of a 2D manifold. This property should be used to scale the shape whenever it is set.
		'''

	@property
	@abstractmethod
	def centroid(self) -> tuple[float, float]:
		'''
		This algorithm is used to calculate the geometric centroid of a 2D manifold. This property should be used move the
		shape about the plane whenever it is set.
		'''

	@abstractmethod
	def draw(self, grid_size: int) -> npt.NDArray[np.int8]:
		'''
		This function creates a boolean mask of a manifold on a grid with dimensions R^(grid_size). The input shape is always
		normalised to the domain R^G before being drawn.
		'''

	@abstractmethod
	def isPointInside(self, p: tuple[float, float]) -> bool:
		'''
		Determines if a given point p ∈ P, including boundaries.
		'''

class ShapeSettings(TypedDict, total=False):
	''' Placeholder for custom ShapeSettings. '''
```
</details>

<details>
<summary>Physics</summary>

### Import

```python
from kac_drumset.physics import (
	# methods
	AdditiveSynthesis,
	ChladniPattern,
	circularAmplitudes,
	circularCymatics,
	circularSeries,
	equilateralTriangleAmplitudes,
	equilateralTriangleSeries,
	FDTDWaveform1D,
	FDTDWaveform2D,
	raisedCosine,
	raisedTriangle,
	linearAmplitudes,
	linearCymatics,
	linearSeries,
	rectangularAmplitudes,
	rectangularCymatics,
	rectangularSeries,
	# classes
	FDTD_2D
)
```

### Methods

```python
def AdditiveSynthesis(
	F: npt.NDArray[np.float64],
	A: npt.NDArray[np.float64],
	d: float,
	k: float,
	T: int,
) -> npt.NDArray[np.float64]:
	'''
	Create a waveform of a 1 or 2-dimensional material using a physically informed
	representation of additive synthesis.
	input:
		F = frequencies (hertz)
		α = spatial eigenfunction ∈ [-1, 1]
		d = decay ∈ [0, ∞)
		k = sample length (ms)
		T = length of simulation (seconds)
	output:
		W[t] = ∑ e^dt * sin(f_n 2πkt) * α_n
		W[t] = ∑ e^dt * sin(f_mn 2πkt) * α_mn
	'''

def ChladniPattern(U: npt.NDArray[np.float64], tolerance: float = 0.1) -> npt.NDArray[np.int8]:
	'''
	Produce a Chladni pattern from a 1 or 2-dimensional cymatic diagram.
	input:
		U = spatial eigenfunction ∈ [-1, 1]
		tolerance = thickness-dependent of the nodal lines
	output:
		B_x = abs(U_x) ≈ 0
		B_xy = abs(U_xy) ≈ 0
	'''

def circularAmplitudes(r: float, theta: float, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
	'''
	Calculate the spatial eigenfunction of a circular 2-dimensional domain relative to a
	polar excitation. The boundary conditions for this spatial eigenfunction are
	determined by the input series of wavenumbers λ_mn.
	input:
		(r, θ) = polar excitation
		S = { λ_mn | λ ∈ ℝ }
	output:
		α_mn = {
			J_m(λ_mn * r * √π) * e^(imθ)
			| α ∈ ℝ, m ∈ [0, M), n ∈ [1, N]
		}
	'''

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
			J_n(z_nm * r) * e^(imθ)
			| U ∈ ℝ^2
		}
	'''

def circularSeries(M: int, N: int, boundary_conditions: bool = True) -> npt.NDArray[np.float64]:
	'''
	Calculate the wavenumbers of a 2-dimensional circular domain.
	input:
		M = number of modes across the Mth axis
		N = number of modes across the Nth axis
		boundary_conditions = (true = fixed, false = free)
	output:
		z_mn = {
			J_m(z_mn) = 0 					dirichlet boundary condition
			J'_m(z_mn) = 0 					neumann boundary condition
			| m ∈ [0, M), n ∈ [1, N]
		}
		λ_mn { z_mn / √π | λ ∈ ℝ }
	'''

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

def linearAmplitudes(
	x: float,
	N: int,
	boundary_conditions: tuple[bool, bool] = (True, True),
) -> npt.NDArray[np.float64]:
	'''
	Calculate the spatial eigenfunction of a 1-dimensional domain relative to a excitation
	location.
	input:
		x = excitation location
		N = number of modes
		boundary_conditions = boolean array indicating the boundary conditions
			(true = fixed, false = free)
			1: x-axis minima boundary condition
			2: x-axis maxima boundary condition
	output:
		α_n = {
			sin((n + 1)πx),			dirichlet boundary condition
			cos(nπx),				neumann boundary condition
			sin((n + 0.5)πx),		minima fixed, maxima free
			cos((n + 0.5)πx),		minima free, maxima fixed
			| α ∈ ℝ, n ∈ [0, N)
		}
	'''

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
			sin((n + 1) πx/X),		dirichlet boundary condition
			cos(nπx/X),				neumann boundary condition
			sin((n + 0.5)πx/X),		minima fixed, maxima free
			cos((n + 0.5)πx/X),		minima free, maxima fixed
			| U ∈ ℝ^1, n ∈ [0, ∞)
		}
	'''

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

def rectangularAmplitudes(
	x: float,
	y: float,
	M: int,
	N: int,
	boundary_conditions: tuple[bool, bool, bool, bool] = (True, True, True, True),
) -> npt.NDArray[np.float64]:
	'''
	Calculate the spatial eigenfunction of a rectangular 2-dimensional domain relative to a
	cartesian excitation.
	input:
		(x, y) = normalised cartesian excitation where [0, 1] represents the mappings onto
			an aspect ratio Є
			x ∈ [0, 1] -> x ∈ [1, √Є]
			y ∈ [0, 1] -> y ∈ [1, 1 / √Є]
		M = number of modes across the Mth axis
		N = number of modes across the Nth axis
		boundary_conditions = boolean array indicating the boundary conditions
			(true = fixed, false = free)
			1: x-axis minima boundary condition
			2: x-axis maxima boundary condition
			3: y-axis minima boundary condition
			4: y-axis maxima boundary condition
	output:
		X_m = {
			sin((m + 1)xπ),		dirichlet boundary condition
			cos(mxπ),			neumann boundary condition
			sin((m + 0.5)xπ),	minima fixed, maxima free
			cos((m + 0.5)xπ),	minima free, maxima fixed
			| m ∈ [0, M)
		}
		Y_n = {
			sin((n + 1)yπ),		dirichlet boundary condition
			cos(nyπ),			neumann boundary condition
			sin((n + 0.5)yπ),	minima fixed, maxima free
			cos((m + 0.5)yπ),	minima free, maxima fixed
			| n ∈ [0, N)
		}
		α_mn = { X_m * Y_n | α ∈ ℝ }
	'''

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
			sin((m + 0.5)xπ / X),	minima fixed, maxima free
			cos((m + 0.5)xπ / X),	minima free, maxima fixed
			| m ∈ [0, ∞)
		}
		Y_n = {
			sin((n + 1)yπ / Y),		dirichlet boundary condition
			cos(nyπ / Y),			neumann boundary condition
			sin((n + 0.5)yπ / Y),	minima fixed, maxima free
			cos((n + 0.5)yπ / Y),	minima free, maxima fixed
			| n ∈ [0, ∞)
		}
		U_xy = { X_m * Y_n | U ∈ ℝ^2 }
	'''

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
```

### Classes

```python
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
	
	def __iter__(self) -> 'FDTD_2D':
		''' Return the iterator. '''

	def __next__(self) -> npt.NDArray[np.float64]:
		''' Compute the FDTD update equation at every iteration. '''
```

</details>

<details><summary>Samplers</summary>

### Import

```python
from kac_drumset.samplers import (
	BesselModel,
	FDTDModel,
	LaméModel,
	LinearFDTD,
	LinearModel,
	PoissonModel,
)
```

### Classes

```python
class BesselModel(AudioSampler):
	'''
	A linear model of a circular membrane using bessel equations of the first kind.
	'''

	class Settings(SamplerSettings, total=False):
		amplitude: float			# maximum amplitude of the simulation ∈ [0, 1]
		boundary_conditions: bool	# control which boundaries are fixed (true) or free (false)
		decay_time: float			# how long will the simulation take to decay? (seconds)
		material_density: float		# material density of the simulated drum membrane (kg/m^2)
		M: int						# number of mth modes
		N: int						# number of nth modes
		tension: float				# tension at rest (N/m)

class FDTDModel(AudioSampler):
	'''
	This class creates a 2D simulation of an arbitrarily shaped drum, calculated using a FDTD scheme.
	'''

	class Settings(SamplerSettings, total=False):
		amplitude: float				# maximum amplitude of the simulation ∈ [0, 1]
		arbitrary_shape: type[Shape]	# what shape should the drum be in?
		decay_time: float				# how long will the simulation take to decay? (seconds)
		drum_size: float				# size of the drum, spanning both the horizontal and vertical axes (m)
		material_density: float			# material density of the simulated drum membrane (kg/m^2)
		shape_settings: ShapeSettings	# the class generator settings for a given drum shape
		strike_width: float				# width of the drum strike (m)
		tension: float					# tension at rest (N/m)

class LaméModel(AudioSampler):
	'''
	A linear model of an equilateral triangle membrane using Lamé equations.
	'''

	class Settings(SamplerSettings, total=False):
		M: int						# number of mth modes
		N: int						# number of nth modes
		amplitude: float			# maximum amplitude of the simulation ∈ [0, 1]
		decay_time: float			# how long will the simulation take to decay? (seconds)
		material_density: float		# material density of the simulated drum membrane (kg/m^2)
		tension: float				# tension at rest (N/m)

class LinearFDTD(AudioSampler):
	'''
	This class creates a 1D simulation, calculated using a FDTD scheme.
	'''

	class Settings(SamplerSettings, total=False):
		'''
		This is an abstract TypedDict used to mirror the type declaration for the customised __init__() method. This allows
		for type safety when using a custom AudioSampler with an arbitrary __init__() method.
		'''

		amplitude: float				# maximum amplitude of the simulation ∈ [0, 1]
		decay_time: float				# how long will the simulation take to decay? (seconds)
		material_density: float			# material density of the simulated drum membrane (kg/m^2)
		strike_width: float				# width of the drum strike (m)
		tension: float					# tension at rest (N/m)

class LinearModel(AudioSampler):
	'''
	A linear model of a string or vibrating air column.
	'''

	class Settings(SamplerSettings, total=False):
		amplitude: float						# maximum amplitude of the simulation ∈ [0, 1]
		boundary_conditions: tuple[bool, bool]	# control which boundaries are fixed (true) or free (false)
		decay_time: float						# how long will the simulation take to decay? (seconds)
		N: int									# number of nth modes
		material_density: float					# material density of the simulated drum membrane (kg/m^2)
		tension: float							# tension at rest (N/m)

class PoissonModel(AudioSampler):
	'''
	A linear model of a unit area rectangle with aspect ratio Є, using poisson equations of the first kind.
	'''

	class Settings(SamplerSettings, total=False):
		amplitude: float									# maximum amplitude of the simulation ∈ [0, 1]
		boundary_conditions: tuple[bool, bool, bool, bool]	# control which boundaries are fixed (true) or free (false)
		decay_time: float									# how long will the simulation take to decay? (seconds)
		M: int												# number of mth modes
		N: int												# number of nth modes
		material_density: float								# material density of the simulated drum membrane (kg/m^2)
		tension: float										# tension at rest (N/m)
```
</details>

# Development

### Dependencies

-   [pipenv](https://formulae.brew.sh/formula/pipenv#default)

### Install

```bash
git clone --recursive ...
pipenv install -d
```
### Build C++ Backend

```bash
pipenv run build
```
### Example

```bash
pipenv run start
```
### Update Dependencies

```bash
pipenv update -d
git submodule update --remote
```
### Test

```bash
pipenv run test
```