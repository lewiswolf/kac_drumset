## Install

To install this project:

```bash
pip ...
```

### Dependencies

-   [libsndfile](https://github.com/libsndfile/libsndfile)

<!-- In either case, _pytorch_ is installed automatically, and will work fine for all CPU based usages. However, to configure this application for GPU usage, you must reinstall the appropriate version of _pytorch_ for your machine (which can be found [here](https://pytorch.org/get-started/locally/)) via:

```bash
pipenv run pip install torch==1.8.1+cu102 ...
```

To ensure that the GPU can be fully utilised by this application, make sure to update the _PATH_2_CUDA_ variable in `src/settings.py`, which should point to your installed version of the CUDA SDK. -->

## Core Library

### Dataset

```python
from kac_drumset import (
	InputFeatures,
	SpectrogramSettings,
)
```

```python
class SpectrogramSettings(TypedDict, total=False):
	'''
	These settings deal strictly with the input representations of the data. For FFT, this is calculated using the
	provided n_bins for the number of frequency bins, window_length and hop_length. The mel representation uses the same
	settings as the FFT, with the addition of n_mels, the number of mel frequency bins.
	'''

	hop_length: int				# hop length in samples
	n_bins: int					# number of frequency bins for the spectral density function
	n_mels: int					# number of mel frequency bins (used when INPUT_FEATURES == 'mel')
	window_length: int			# window length in samples


class InputFeatures():
	'''
	This class is used to convert a raw waveform into a user defined input representation, which includes end2end, the
	fourier transform, and a mel spectrogram. The intended use of this class when deployed:
		IF = InputFeatures()
		X = np.zeros((n,) + IF.transformShape(len(waveform))))
		for i in range(n):
			X[i] = IF.transform(waveform)
	'''

	def __init__(
		self,
		feature_type: Literal['end2end', 'fft', 'mel'],
		sr: int,
		normalise_input: bool = True,
		spectrogram_settings: SpectrogramSettings = {},
	) -> None:
		'''
		InputFeatures works by creating a variably defined method self.transform. This method uses the input settings to
		generate the correct input representation of the data.
		'''

	@staticmethod
	def normalise(waveform: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
		'''
		Normalise an audio waveform, such that x ∈ [-1.0, 1.0]
		'''
		
	def transform(self, npt.NDArray[np.float64]) -> torch.Tensor:
		'''
		Return a representation of the input material after applying the audio transform. 
		'''

	def transformShape(self, data_length: int) -> tuple[int, ...]:
		'''
		Helper method used for precomputing the shape of an individual input feature.
		params:
			data_length		Length of the audio file (samples).
		'''
```

### Sampler

```python
from kac_drumset import (
	AudioSampler,
)
```

```python
```

### Geometry

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

## Development

### Dependencies

-   [pipenv](https://formulae.brew.sh/formula/pipenv#default)
-	[cmake](https://formulae.brew.sh/formula/cmake)
<!-- -   [CUDA SDK](https://developer.nvidia.com/cuda-downloads) -->

### Install

```bash
git clone ...
pipenv install -d
```

### Build 

```bash
pipenv run build
```

### Test

```
pipenv run test
```

### Helper Methods

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