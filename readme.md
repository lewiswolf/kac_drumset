<!-- ![Kac-Drumming](https://user-images.githubusercontent.com/55607290/169860844-7f3f3d6d-4366-4410-8a30-5ee9472c2864.png) -->

# kac_drumset

Analysis tools and dataset generator for arbitrarily shaped drums.

# Install

```bash
pip install "git+https://github.com/lewiswolf/kac_drumset.git@dev#egg=kac_drumset"
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
	transformDataset,
	# Classes
	AudioSampler,
	InputRepresentation,
	# Types
	RepresentationSettings,
	SamplerSettings,
	TorchDataset,
)
```

### Methods

```python
def generateDataset(
	Sampler: Type[AudioSampler],
	sampler_settings: SamplerSettings,
	dataset_dir: str = os.path.normpath(f'{os.path.dirname(__file__)}/../../data'),
	dataset_size: int = 10,
	representation_settings: RepresentationSettings = {},
) -> TorchDataset:
	'''
	Generates a dataset of audio samples. The generated dataset, including the individual .wav files and the metadata.json,
	are saved in the directory specified by the absolute filepath dataset_dir.
	'''


def loadDataset(dataset_dir: str = os.path.normpath(f'{os.path.dirname(__file__)}/../../data')) -> TorchDataset:
	'''
	loadDataset imports a kac_drumset dataset from the directory specified by the absolute path dataset_dir.
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
	def getLabels(self) -> list[Union[float, int]]:
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
	sampler: str										# the name of the sampler used to generate the dataset
	sampler_settings: SamplerSettings					# settings for the sampler
	X: torch.Tensor										# data
	Y: torch.Tensor										# labels

	def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
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
	'area',
	'booleanMask',
	'centroid',
	'generateConcave',
	'generateConvexPolygon',
	'groupNormalisation',
	'isColinear',
	'isConvex',
	'largestVector',
	# Classes
	'RandomPolygon',
	# Types
	'Polygon',
)
```

### Methods

```python
def area(vertices: npt.NDArray[np.float64]) -> float:
	'''
	An implementation of the shoelace algorithm, first described by Albrecht Ludwig
	Friedrich Meister, which is used to calculate the area of a polygon. The area
	of a polygon can also be computed (using Green's theorem directly) using
	`cv2.contourArea(self.vertices.astype('float32'))`. However, this function
	requires that the input be of the type float32, resulting in a trade off between
	(marginal) performance gains and lower precision.
	'''

def booleanMask(
	vertices: npt.NDArray[np.float64],
	grid_size: int,
	convex: Optional[bool],
) -> npt.NDArray[np.int8]:
	'''
	This function creates a boolean mask of an input polygon on a grid with dimensions
	R^(grid_size). The input shape should exist within a domain R^G where G ∈ [0, 1].
	'''

def centroid(vertices: npt.NDArray[np.float64], area: float) -> tuple[float, float]:
	'''
	This algorithm is used to calculate the geometric centroid of a 2D polygon.
	See http://paulbourke.net/geometry/polygonmesh/ 'Calculating the area and
	centroid of a polygon'.
	'''

def generateConcave(n: int) -> npt.NDArray[np.float64]:
	'''
	Generates a random concave shape, with a small probability of also returning a
	convex shape. It should be noted that this function can not be used to create
	all possible simple polygons; see todo.md => 'Missing a reliable algorithm to
	generate all concave shapes'.
	'''

def generateConvex(n: int) -> npt.NDArray[np.float64]:
	'''
	Generate convex shapes according to Pavel Valtr's 1995 algorithm. Adapted
	from Sander Verdonschot's Java version, found here:
		https://cglab.ca/~sander/misc/ConvexGeneration/ValtrAlgorithm.java
	'''

def groupNormalisation(
	vertices: npt.NDArray[np.float64],
	convex: Optional[bool],
) -> npt.NDArray[np.float64]:
	'''
	This function uses the largest vector to define a polygon's span across the
	y-axis. After finding the largest vector, the polygon is rotated about said
	vector's midpoint, and finally the entire polygon is normalised to span the
	unit interval.
	'''

def isColinear(vertices: npt.NDArray[np.float64]) -> bool:
	'''
	Determines whether or not a given set of three vertices are colinear.
	'''

def isConvex(vertices: npt.NDArray[np.float64]) -> bool:
	'''
	Tests whether or not a given array of vertices forms a convex polygon. This is
	achieved using the resultant sign of the cross product for each vertex:
		[(x_i - x_i-1), (y_i - y_i-1)] x [(x_i+1 - x_i), (y_i+1 - y_i)]
	See => http://paulbourke.net/geometry/polygonmesh/ 'Determining whether or not
	a polygon (2D) has its vertices ordered clockwise or counter-clockwise'.
	'''

def largestVector(vertices: npt.NDArray[np.float64]) -> tuple[float, tuple[int, int]]:
	'''
	This function tests each pair of vertices in a given polygon to find the largest
	vector, and returns the length of the vector and its indices.
	'''
```

### Classes

```python
class RandomPolygon(Polygon):
	'''
	This class is used to generate a random polygon, normalised and centred between 0.0
	and 1.0. The area and the centroid of the polygon are also included in this class.
	'''

	area: float							# area of the polygon
	centroid: tuple[float, float]		# coordinate pair representing the centroid of the polygon
	convex: bool						# is the polygon convex?

	def __init__(self, max_vertices: int, allow_concave: bool = False) -> None:
		'''
		This function generates a polygon, whilst also calculating its properties.
		input:
			max_vertices:	Maximum amount of vertices. The true value is a uniform distribution from 3 to max_vertices.
			allow_concave:	Is this polygon allowed to be concave?
		'''
```

### Types

```python
class Polygon():
	'''
	A base class for a polygon, instantiated with an array of vertices.
	'''

	n: int								# number of vertices
	vertices: npt.NDArray[np.float64]	# cartesian products representing the vertices of a shape
```
</details>

<details>
<summary>Physics</summary>

### Import

```python
from kac_drumset.physics import (
	FDTDWaveform2D,
	raisedCosine,
)
```

### Methods

```python
def FDTDWaveform2D(
	u_0: npt.NDArray[np.float64],
	u_1: npt.NDArray[np.float64],
	B: npt.NDArray[np.int8],
	c_0: float,
	c_1: float,
	d: float,
	T: int,
	x_range: tuple[int, int],
	y_range: tuple[int, int],
	w: tuple[float, float],
) -> npt.NDArray[np.float64]:
	'''
	Generates a waveform using a 2 dimensional FDTD scheme.
	input:
		u_0 = initial fdtd grid at t = 0.
		u_1 = initial fdtd grid at t = 1.
		B = boundary conditions.
		c_0 = first fdtd coefficient related to the courant number.
		c_1 = second fdtd coefficient related to the courant number.
		d = decay coefficient.
		T = length of simulation in samples.
		x_range = tuple representing the range across the x axis to update.
		y_range = tuple representing the range across the y axis to update.
		w = the coordinate at which the waveform is sampled.
	output:
		waveform = W[n] ∈
			(λ ** 2)(
				u_n_x+1_y + u_n_x-1_y + u_n_x_y+1 + u_n_x_y-1
			) + 2(1 - 2(λ ** 2))u_n_x_y - d(u_n-1_x_y) ∀ u ∈ R^2
	'''

def raisedCosine(
	matrix_size: tuple[int, ...],
	mu: tuple[int, ...],
	sigma: float = 0.5,
) -> npt.NDArray[np.float64]:
	'''
	This function creates a raised cosine distribution centred at mu. Only 1D and 2D distributions are supported.
	input:
		matrix_size		A tuple representing the size of the output matrix.
		mu				The coordinate used to represent the centre of the
						cosine distribution.
		sigma			The radius of the distribution.
	'''
```

</details>

<details><summary>Samplers</summary>

### Import

```python
from kac_drumset import (
	FDTDModel,
)
```

### Classes

```python
class FDTDModel(AudioSampler):
	'''
	This class creates a 2D simulation of an arbitrarily shaped drum, calculated
	using a FDTD scheme.
	'''

	class Settings(SamplerSettings, total=False):
		'''
		This is an abstract TypedDict used to mirror the type declaration for the customised __init__() method. This allows
		for type safety when using a custom AudioSampler with an arbitrary __init__() method.
		'''

		amplitude: float			# maximum amplitude of the simulation ∈ [0, 1]
		decay_time: float			# how long will the simulation take to decay?
		drum_size: float			# size of the drum, spanning both the horizontal and vertical axes (m)
		material_density: float		# material density of the simulated drum membrane (kg/m^2)
		max_vertices: int			# maximum amount of vertices for a given drum
		tension: float				# tension at rest (N/m)
```
</details>

# Development

### Dependencies

-   [pipenv](https://formulae.brew.sh/formula/pipenv#default)

### Install

```bash
git clone ...
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
	f_0: float										# fundamental frequency (hz)
	waveshape: Literal['saw', 'sin', 'sqr', 'tri']	# shape of the waveform
```
</details>
