<!-- ![Kac-Drumming](https://user-images.githubusercontent.com/55607290/169860844-7f3f3d6d-4366-4410-8a30-5ee9472c2864.png) -->

# kac_drumset

![python version](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue)
<a href="https://doi.org/10.5281/zenodo.7274474">
![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.7274474-blue)
</a>

Python based analysis tools and dataset generator for arbitrarily shaped drums.

# Install

```bash
pip install "git+https://github.com/lewiswolf/kac_drumset.git#egg=kac_drumset"
```

### Dependencies

-	[cmake](https://formulae.brew.sh/formula/cmake)
-   [libsndfile](https://github.com/libsndfile/libsndfile)

# Core Library

<details>
<summary>Dataset</summary>

### Import

```python
from kac_drumset import (
	# Methods
	generateDataset,
	loadDataset,
	regenerateDataPoints,
	transformDataset,
	# Classes
	AudioSampler,
	InputRepresentation,
	# Types
	RepresentationSettings,
	SamplerInfo,
	SamplerSettings,
	TorchDataset,
)
```

### Methods

```python
def generateDataset(
	Sampler: Type[AudioSampler],
	sampler_settings: SamplerSettings,
	dataset_dir: str,
	dataset_size: int = 10,
	representation_settings: RepresentationSettings = {},
) -> TorchDataset:
	'''
	Generates a dataset of audio samples. The generated dataset, including the individual .wav files and the metadata.json,
	are saved in the directory specified by the absolute filepath dataset_dir.
	'''

def loadDataset(dataset_dir: str) -> TorchDataset:
	'''
	loadDataset imports a kac_drumset dataset from the directory specified by the absolute path dataset_dir.
	'''

def regenerateDataPoints(dataset: TorchDataset, Sampler: type[AudioSampler], entries: list[int]) -> TorchDataset:
	'''
	This method regenerates specific indices of a dataset.
	'''

def transformDataset(dataset: TorchDataset, representation_settings: RepresentationSettings) -> TorchDataset:
	'''
	transformDataset is used to transform the input representation of a loaded dataset. This method rewrites the
	metadata.json for the dataset, such that the dataset will be loaded with the new settings upon future use.
	'''
```

### Classes

```python
class AudioSampler(ABC):
	''' Abstract parent class for an audio sampler. '''

	duration: float						# duration of the audio file (seconds)
	length: int							# length of the audio file (samples)
	sample_rate: int					# sample rate
	waveform: npt.NDArray[np.float64]	# the audio sample itself

	def export(self, absolutePath: str, bit_depth: Literal[16, 24, 32] = 24) -> None:
		''' Write the generated waveform to a .wav file. '''

	@abstractmethod
	def generateWaveform(self) -> None:
		''' This method should be used to generate and set self.waveform. '''

	@abstractmethod
	def getLabels(self) -> dict[str, list[Union[float, int]]]:
		''' This method should return the y labels for the generated audio. '''

	@abstractmethod
	def updateProperties(self, i: Union[int, None]) -> None:
		''' This method should be used to update the properties of the sampler when inside a generator loop. '''

	@abstractmethod
	class Settings(SamplerSettings, total=False):
		'''
		This is an abstract TypedDict used to mirror the type declaration for the customised __init__() method. This allows
		for type safety when using a custom AudioSampler with an arbitrary __init__() method.
		'''

class InputRepresentation():
	'''
	This class is used to convert a raw waveform into a user defined input representation, which includes end2end, the
	fourier transform, and a mel spectrogram.
	'''

	settings: RepresentationSettings

	def __init__(self, sample_rate: int, settings: RepresentationSettings = {}) -> None:
		'''
		InputRepresentation works by creating a variably defined method self.transform. This method uses the input settings to
		generate the correct input representation of the data.
		'''

	def transform(self, waveform: npt.NDArray[np.float64]) -> torch.Tensor:
		''' Produce the output representation. '''

	@staticmethod
	def normalise(waveform: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
		''' Normalise an audio waveform, such that x ∈ [-1.0, 1.0] '''

	@staticmethod
	def transformShape(data_length: int, settings: RepresentationSettings) -> tuple[int, ...]:
		''' This method uses the length of the incoming audio data to calculate the size of the transform's output. '''
```

### Types

```python
class RepresentationSettings(TypedDict, total=False):
	'''
	These settings are used to specify the data representation of audio, providing the option for end to end data, as well
	as Fourier and Mel transformations. An FFT is calculated using n_bins for the number of frequency bins, as well as
	window_length and hop_length for the size of the bins. The Mel representation uses the same settings as the FFT, with
	the addition of n_mels, the number of mel frequency bins, and f_min, the minimum frequency of the transform.
	'''

	f_min: float			# minimum frequency of the transform in hertz (mel only)
	hop_length: int			# hop length in samples
	n_bins: int				# number of frequency bins for the spectral density function
	n_mels: int				# number of mel frequency bins (mel only)
	normalise_input: bool	# should the input be normalised
	output_type: Literal[	# representation type
		'end2end',
		'fft',
		'mel',
	]
	window_length: int		# window length in samples

class SamplerInfo(TypedDict, total=True):
	'''
	Information about the sampler used to generate a specific dataset.
	'''
	name: str		# name of the sampler
	version: str	# version of kac_drumset when the sampler was generated

class SamplerSettings(TypedDict, total=True):
	'''
	These are the minimum requirements for the AudioSampler __init__() method. This type is used to maintain type safety
	when using a custom AudioSampler.
	'''
	duration: float		# duration of the audio file (seconds)
	sample_rate: int	# sample rate

class TorchDataset(torch.utils.data.Dataset):
	''' PyTorch wrapper for a dataset. '''

	dataset_dir: str									# dataset directory
	representation_settings: RepresentationSettings		# settings for InputRepresentation
	sampler: SamplerInfo								# the name of the sampler used to generate the dataset
	sampler_settings: dict[str, Any]					# settings for the sampler
	X: torch.Tensor										# data
	Y: list[dict[str, torch.Tensor]]					# labels

	def __getitem__(self, i: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
		''' Return the data and its labels at index i. '''

	def __len__(self) -> int:
		''' Return the dataset size. '''
```
</details>

<details>
<summary>Geometry</summary>

### Import

```python
from kac_drumset.geometry import (
	# Methods
	centroid,
	convexNormalisation,
	drawCircle,
	drawPolygon,
	generateConvexPolygon,
	isColinear,
	isPointInsidePolygon,
	largestVector,
	weylCondition
	# Classes
	RandomPolygon,
	UnitRectangle,
	# Types
	Circle,
	Polygon,
	Shape,
)
```

### Methods

```python
def centroid(P: Polygon) -> tuple[float, float]:
	'''
	This algorithm is used to calculate the geometric centroid of a 2D polygon. 
	See http://paulbourke.net/geometry/polygonmesh/ 'Calculating the area and centroid of a polygon'.
	'''

def convexNormalisation(P: Polygon) -> npt.NDArray[np.float64]:
	'''
	This algorithm produces an identity polygon for each unique polygon given as input. This method normalises an input
	polygon to the unit interval such that x ∈ [0, 1] && y ∈ [0, 1], reducing each input polygon by isometric and
	similarity transformations. This is achieved by first enforcing that the vertices of a polygon are ordered clockwise.
	Then, the largest vector is used to determine the lower and upper bounds across the x-axis. Next, the polygon is split
	into quadrants, the largest of whose area determines the rotation/reflection of the polygon. Finally, the points are
	normalised, and ordered such that V[0] = [0., y].
	'''

def drawCircle(C: Circle, grid_size: int) -> npt.NDArray[np.int8]:
	'''
	This function creates a boolean mask of a circle on a grid with dimensions R^(grid_size). The input shape should
	exist within a domain R^G where G ∈ [0, 1].
	'''

def drawPolygon(P: Polygon, grid_size: int) -> npt.NDArray[np.int8]:
	'''
	This function creates a boolean mask of a polygon on a grid with dimensions R^(grid_size). The input shape should
	exist within a domain R^G where G ∈ [0, 1].
	'''

def generateConvexPolygon(N: int) -> npt.NDArray[np.float64]:
	'''
	Generate convex shapes according to Pavel Valtr's 1995 algorithm. Adapted from Sander Verdonschot's Java version,
	found here: https://cglab.ca/~sander/misc/ConvexGeneration/ValtrAlgorithm.java
	'''

def isColinear(vertices: npt.NDArray[np.float64]) -> bool:
	'''
	Determines whether or not a given set of three vertices are colinear.
	'''

def isPointInsidePolygon(p: tuple[float, float], P: Polygon) -> bool:
	'''
	Determines whether or not a cartesian pair is within a polygon, including boundaries.
	'''

def largestVector(P: Polygon) -> tuple[float, tuple[int, int]]):
	'''
	This function tests each pair of vertices in a given polygon to find the largest vector, and returns the length of the
	vector and its indices.
	'''

def weylCondition(S_1: Shape, S_2: Shape) -> bool:
	'''
	Using Weyl's asymptotic law, determine whether two polygons may be isospectral.
	https://en.wikipedia.org/wiki/Weyl_law
	'''
```

### Classes

```python
class RandomPolygon(Polygon):
	'''
	This class is used to generate a random polygon, normalised and centred between 0.0 and 1.0. The convexity and the
	centroid of the polygon are also included in this class.
	'''

	centroid: tuple[float, float]		# coordinate pair representing the centroid of the polygon
	convex: bool						# is the polygon convex?

	def __init__(self, max_vertices: int, allow_concave: bool = False) -> None:
		'''
		This function generates a polygon, whilst also calculating its properties.
		input:
			max_vertices:	Maximum amount of vertices. The true value is a uniform distribution from 3 to max_vertices.
			allow_concave:	Is this polygon allowed to be concave?
		'''

class UnitRectangle(Polygon):
	'''
	Define the unit rectangle.
	'''

	def __init__(self, epsilon: float = 1.) -> None:
		'''
		input:
			epsilon = aspect ratio
		'''
```

### Types

```python
class Circle(Shape):
	'''
	A base class for a circle, instantiated with a radius.
	'''

	r: float 							# radius

	def __init__(self, r: float = 1.) -> None:
		'''
		input:
			r = radius
		'''

	@cached_property
	def area(self) -> float:
		''' Archimedes. '''

class Polygon(Shape):
	'''
	A base class for a polygon, instantiated with an array of vertices.
	'''

	N: int								# number of vertices
	vertices: npt.NDArray[np.float64]	# cartesian products representing the vertices of a shape

	def __init__(self, vertices: Union[list[list[float]], npt.NDArray[np.float64]]) -> None:
		'''
		input:
			vertices = array of cartesian points.
		'''

	@cached_property
	def area(self) -> float:
		'''
		An implementation of the shoelace algorithm, first described by Albrecht Ludwig Friedrich Meister, which is used to
		calculate the area of a polygon.
		'''
	
	@cached_property
	def convex(self) -> bool:
		'''
		Tests whether or not a given polygon is convex. This is achieved using the resultant sign of the cross product for
		each vertex:
			[(x_i - x_i-1), (y_i - y_i-1)] × [(x_i+1 - x_i), (y_i+1 - y_i)].
		See => http://paulbourke.net/geometry/polygonmesh/ 'Determining whether or not a polygon (2D) has its vertices ordered
		clockwise or counter-clockwise'.
		'''

class Shape(ABC):
	'''
	An abstract base class for a shape in Euclidean geometry.
	'''
	
	@abstractmethod
	@cached_property
	def area(self) -> float:
		pass
```
</details>

<details>
<summary>Physics</summary>

### Import

```python
from kac_drumset.physics import (
	# methods
	besselJ,
	besselJZero,
	calculateCircularAmplitudes,
	calculateCircularSeries,
	calculateRectangularAmplitudes,
	calculateRectangularSeries,
	FDTDWaveform2D,
	raisedCosine,
	raisedTriangle,
	# classes
	FDTD_2D
)
```

### Methods

```python
def besselJ(n: float, m: float) -> float:
	'''
	Calculate the bessel function of the first kind. This method is a clone of boost::math::cyl_bessel_j.
	'''

def besselJZero(n: float, m: int) -> float:
	'''
	Calculate the mth zero crossing of the nth bessel function of the first kind. This method is a clone of
	boost::math::cyl_bessel_j_zero.
	'''

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

def calculateCircularSeries(N: int, M: int) -> npt.NDArray[np.float64]:
	'''
	Calculate the eigenmodes of a circle.
	input:
		N = number of modal orders
		M = number of modes per order
	output:
		S = { z_nm | s ∈ ℝ, J_n(z_nm) = 0, n < N, 0 < m <= M }
	'''

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

def FDTDWaveform2D(
	u_0: npt.NDArray[np.float64],
	u_1: npt.NDArray[np.float64],
	B: npt.NDArray[np.int8],
	c_0: float,
	c_1: float,
	c_2: float,
	T: int,
	w: tuple[int, int],
) -> npt.NDArray[np.float64]:
	'''
	Generates a waveform using a 2 dimensional FDTD scheme.
	input:
		u_0 = initial fdtd grid at t = 0.
		u_1 = initial fdtd grid at t = 1.
		B = boundary conditions.
		c_0 = first fdtd coefficient related to the decay term and the courant number.
		c_1 = second fdtd coefficient related to the decay term and the courant number.
		c_2 = third fdtd coefficient related to the decay term.
		T = length of simulation in samples.
		w = the coordinate at which the waveform is sampled.
	output:
		waveform = W[n] ∈
			c_0 * (
				u_n_x+1_y + u_n_x-1_y + u_n_x_y+1 + u_n_x_y-1
			) + c_1 * u_n_x_y - c_2 * (u_n-1_x_y) ∀ u ∈ R^2
	'''

def raisedCosine(
	matrix_size: tuple[int, ...],
	mu: tuple[float, ...],
	sigma: float = 0.5,
) -> npt.NDArray[np.float64]:
	'''
	Creates a raised cosine distribution centred at mu. Only 1D and 2D distributions are supported.
	input:
		matrix_size = A tuple representing the size of the output matrix.
		μ = The coordinate used to represent the centre of the cosine distribution.
		σ = The radius of the distribution.
	'''

def raisedTriangle(matrix_size: int, mu: float, a: float, b: float) -> npt.NDArray[np.float64]:
	'''
	Create a triangular distribution centred at mu. Only 1D distributions are supported.
	input:
		matrix_size = the size of the matrix.
		μ = a cartesian point representing the maxima of the triangle.
		a = minimum x value for the distribution.
		b = maximum x value for the distribution.
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
from kac_drumset import (
	BesselModel,
	FDTDModel,
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
		M: int						# number of mth modes
		N: int						# number of nth modes
		amplitude: float			# maximum amplitude of the simulation ∈ [0, 1]
		decay_time: float			# how long will the simulation take to decay? (seconds)
		material_density: float		# material density of the simulated drum membrane (kg/m^2)
		tension: float				# tension at rest (N/m)

class FDTDModel(AudioSampler):
	'''
	This class creates a 2D simulation of an arbitrarily shaped drum, calculated using a FDTD scheme.
	'''

	class Settings(SamplerSettings, total=False):
		amplitude: float			# maximum amplitude of the simulation ∈ [0, 1]
		decay_time: float			# how long will the simulation take to decay? (seconds)
		drum_size: float			# size of the drum, spanning both the horizontal and vertical axes (m)
		material_density: float		# material density of the simulated drum membrane (kg/m^2)
		max_vertices: int			# maximum amount of vertices for a given drum
		strike_width: float			# width of the drum strike (m)
		tension: float				# tension at rest (N/m)

class PoissonModel(AudioSampler):
	'''
	A linear model of a unit area rectangle with aspect ratio Є, using poisson equations of the first kind.
	'''

	class Settings(SamplerSettings, total=False):
		M: int						# number of mth modes
		N: int						# number of nth modes
		amplitude: float			# maximum amplitude of the simulation ∈ [0, 1]
		decay_time: float			# how long will the simulation take to decay? (seconds)
		material_density: float		# material density of the simulated drum membrane (kg/m^2)
		tension: float				# tension at rest (N/m)
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
### Build 

```bash
pipenv run build
```
### Example

```
pipenv run start
```
### Test

```
pipenv run test
```

<details><summary>Testing Library</summary>

### Import

```python
# Methods
from kac_drumset.utils import (
	withoutPrinting,
	withProfiler,
	withTimer,
)
# Samplers
from kac_drumset import (
	TestSweep,
	TestTone,
)
```

### Methods

```python
def withoutPrinting(allow_errors: bool = False) -> Iterator[Any]:
	'''
	This wrapper can used around blocks of code to silence calls to print(), as well as optionally silence error messages.
	'''

def withProfiler(func: Callable, n: int, *args: Any, **kwargs: Any) -> None:
	'''
	Calls the input function using cProfile to generate a performance report in the console. Prints the n most costly functions.
	'''

def withTimer(func: Callable, *args: Any, **kwargs: Any) -> None:
	'''
	Calls the input function and posts its runtime to the console.
	'''
```

### Samplers

```python
class TestSweep(AudioSampler):
	'''
	This class produces a sine wave sweep across the audio spectrum, from 20hz to f_s / 2.
	'''
		
class TestTone(AudioSampler):
	'''
	This class produces an arbitrary test tone, using either a sawtooth, sine, square or triangle waveform. If it's initial frequency is not set, it will automatically create random frequencies.
	'''

	class Settings(SamplerSettings, total=False):
		f_0: float										# fixed fundamental frequency (hz)
		waveshape: Literal['saw', 'sin', 'sqr', 'tri']	# shape of the waveform
```
</details>
