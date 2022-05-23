# Install

To install this project:

```bash
pip ...
```

## Dependencies

-   [libsndfile](https://github.com/libsndfile/libsndfile)

<!-- In either case, _pytorch_ is installed automatically, and will work fine for all CPU based usages. However, to configure this application for GPU usage, you must reinstall the appropriate version of _pytorch_ for your machine (which can be found [here](https://pytorch.org/get-started/locally/)) via:

```bash
pipenv run pip install torch==1.8.1+cu102 ...
```

To ensure that the GPU can be fully utilised by this application, make sure to update the _PATH_2_CUDA_ variable in `src/settings.py`, which should point to your installed version of the CUDA SDK. -->

# Core Library

<details><summary>Dataset</summary>

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

### Classes

```python
class AudioSampler(ABC):
	'''
	Abstract parent class for an audio sampler. The intended use when deployed:

	sampler = AudioSampler()
	for i in range(N):
		sampler.updateParameters(i)
		sampler.generateWaveform()
		x = sampler.waveform
		y = sampler.getLabels()
		sampler.export('/absolute/filepath/')
	'''

	def __init__(self, duration: float, sample_rate: int) -> None:
		''' Initialise sampler. '''

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
```

### Types

```python
class SamplerSettings(TypedDict, total=True):
	'''
	These are the minimum requirements for the AudioSampler __init__() method. This type is used to maintain type safety
	when using a custom AudioSampler.
	'''
	duration: float
	sample_rate: int
```
</details>

<details><summary>Samplers</summary>
</details>

<details><summary>Geometry</summary>

```python
import kac_drumset.geometry as G
```

```python
class Polygon():
	'''
	A base class for a polygon, instantiated with an array of vertices.
	'''

class RandomPolygon(Polygon):
	'''
	This class is used to generate a random polygon, normalised and centred between 0.0
	and 1.0. The area and the centroid of the polygon are also included in this class.
	'''

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
</details>

# Development

## Dependencies

-   [pipenv](https://formulae.brew.sh/formula/pipenv#default)
-	[cmake](https://formulae.brew.sh/formula/cmake)
<!-- -   [CUDA SDK](https://developer.nvidia.com/cuda-downloads) -->

## Install

```bash
git clone ...
pipenv install -d
```

## Build 

```bash
pipenv run build
```

## Test

```
pipenv run test
```

## Testing Library

<details><summary>Helper Methods</summary>

```python
from kac_drumset import (
	TestSweep,
	TestTone,
)
from kac_drumset.utils import (
	withoutPrinting,
	withProfiler,
	withTimer,
)
```

```python
class TestSweep(AudioSampler):
	'''
	This class produces a sine wave sweep across the audio spectrum, from 20hz to f_s / 2.
	'''
		
class TestTone(AudioSampler):
	'''
	This class produces an arbitrary test tone, using either a sawtooth, sine, square or triangle waveform. If it's initial frequency is not set, it will automatically create random frequencies.
	'''

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
</details>