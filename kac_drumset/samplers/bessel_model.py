'''
This sampler is used to produce a linear model of a circular membrane.
'''

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from kac_prediction.dataset import classLocalsToKwargs, AudioSampler, SamplerSettings
from ..physics import AdditiveSynthesis, circularAmplitudes, circularSeries

__all__ = [
	'BesselModel',
]


class BesselModel(AudioSampler):
	'''
	A linear model of a circular membrane using bessel equations of the first kind.
	'''

	# user defined variables
	a: float						# maximum amplitude of the simulation ∈ [0, 1]
	bc: bool						# control which boundaries are fixed (true) or free (false)
	d_60: float						# decay time (seconds)
	M: int							# number of mth modes
	N: int							# number of nth modes
	p: float						# material density of the simulated drum membrane (kg/m^2)
	t: float						# tension at rest (N/m)
	# model inferences
	c: float						# wavespeed (m/s)
	decay: float					# decay constant
	F: npt.NDArray[np.float64]		# array of frequencies (hz)
	k: float						# sample length (ms)
	series: npt.NDArray[np.float64]	# array of wavenumbers λ_nm
	# drum properties
	L: float						# size of the drum (m)
	strike: tuple[float, float]		# normalised excitation location in polar coordinates

	class Settings(SamplerSettings, total=False):
		'''
		This is an abstract TypedDict used to mirror the type declaration for the customised __init__() method. This allows
		for type safety when using a custom AudioSampler with an arbitrary __init__() method.
		'''

		amplitude: float			# maximum amplitude of the simulation ∈ [0, 1]
		boundary_conditions: bool	# control which boundaries are fixed (true) or free (false)
		decay_time: float			# how long will the simulation take to decay? (seconds)
		material_density: float		# material density of the simulated drum membrane (kg/m^2)
		M: int						# number of mth modes
		N: int						# number of nth modes
		tension: float				# tension at rest (N/m)

	def __init__(
		self,
		duration: float,
		sample_rate: int,
		amplitude: float = 1.,
		boundary_conditions: bool = True,
		decay_time: float = 2.,
		material_density: float = 0.2,
		M: int = 10,
		N: int = 10,
		tension: float = 2000.,
	) -> None:
		'''
		When the class is first instantiated, all of its physical properties are inferred from the user parameters.
		'''

		# initialise user defined variables
		super().__init__(**classLocalsToKwargs(locals()))
		self.a = amplitude
		self.bc = boundary_conditions
		self.d_60 = decay_time
		self.M = M
		self.N = N
		self.p = material_density
		self.t = tension
		# initialise inferences
		self.c = (self.t / self.p) ** 0.5
		self.k = 1. / self.sample_rate
		self.decay = -1 * self.k * 6 * np.log(10) / self.d_60
		self.series = circularSeries(self.M, self.N, self.bc)

	def generateWaveform(self) -> None:
		'''
		Using additive synthesis, generate the waveform for the linear model.
		'''

		self.waveform = AdditiveSynthesis(
			self.F,
			self.a * circularAmplitudes(*self.strike, self.series),
			self.decay,
			self.k,
			self.length,
		)

	def getLabels(self) -> dict[str, list[float | int]]:
		'''
		Return the labels of the bessel model.
		'''

		return {'drum_size': [self.L], 'strike_location': [*self.strike]} if hasattr(self, 'L') else {}

	def updateProperties(self, i: int | None = None) -> None:
		'''
		For every five drum samples generated, update the size of the drum. And for every drum sample generated update the
		strike location - the first strike location is always the centroid.
		'''

		if i is None or i % 5 == 0:
			# initialise a random drum size and strike location in the centroid of the drum.
			self.L = np.random.uniform(0.1, 1.)
			self.F = self.series * self.c / (2. * self.L)
			self.strike = (0., 0.)
		else:
			# otherwise update the strike location to be a random location.
			self.strike = (np.random.uniform(-1., 1.), np.random.uniform(0., np.pi))
