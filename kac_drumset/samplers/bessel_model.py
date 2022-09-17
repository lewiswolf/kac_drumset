'''
This sampler is used to produce a linear model of a circular membrane.
'''

# core
import random
from typing import Union

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from ..dataset import AudioSampler, SamplerSettings
from ..physics import calculateCircularAmplitudes, calculateCircularSeries

__all__ = [
	'BesselModel',
]


class BesselModel(AudioSampler):
	'''
	A linear model of a circular membrane using bessel equations of the first kind.
	'''

	# user defined variables
	a: float						# maximum amplitude of the simulation ∈ [0, 1]
	d_60: float						# decay time (seconds)
	M: int							# number of mth modes
	N: int							# number of nth modes
	p: float						# material density of the simulated drum membrane (kg/m^2)
	t: float						# tension at rest (N/m)
	# model inferences
	c: float						# wavespeed (m/s)
	decay: float					# decay constant
	gamma: float					# scaled wavespeed (1/s)
	k: float						# sample length (ms)
	series: npt.NDArray[np.float64]	# array of eigenmodes, z_nm
	# drum properties
	L: float						# diameter of the drum (m)
	strike: tuple[float, float]		# strike location in cartesian coordinates

	class Settings(SamplerSettings, total=False):
		'''
		This is an abstract TypedDict used to mirror the type declaration for the customised __init__() method. This allows
		for type safety when using a custom AudioSampler with an arbitrary __init__() method.
		'''

		amplitude: float			# maximum amplitude of the simulation ∈ [0, 1]
		decay_time: float			# how long will the simulation take to decay? (seconds)
		M: int						# number of mth modes
		material_density: float		# material density of the simulated drum membrane (kg/m^2)
		N: int						# number of nth modes
		tension: float				# tension at rest (N/m)

	def __init__(
		self,
		duration: float,
		sample_rate: int,
		amplitude: float = 1.,
		decay_time: float = 2.,
		M: int = 10,
		material_density: float = 0.2,
		N: int = 10,
		tension: float = 2000.,
	) -> None:
		'''
		When the class is first instantiated, all of its physical properties are inferred from the user parameters.
		'''

		# initialise user defined variables
		super().__init__(duration, sample_rate)
		self.a = amplitude
		self.d_60 = decay_time
		self.M = M
		self.N = N
		self.p = material_density
		self.t = tension
		# initialise inferences
		self.c = (self.t / self.p) ** 0.5
		self.k = 1. / self.sample_rate
		self.decay = -1 * self.k * 6 * np.log(10) / self.d_60
		self.series = calculateCircularSeries(N, M)

	def generateWaveform(self) -> None:
		'''
		Using additive synthesis, generate the waveform for the linear model.
		'''

		# 2016 - Chaigne & Kergomard, p.154
		omega = (self.gamma * self.series).flatten() # eigenfrequencies
		omega *= 2 * np.pi * self.k # rate of phase
		A = self.a * np.abs(calculateCircularAmplitudes(*self.strike, self.series)).flatten()
		for i in range(self.length):
			# 2009 - Bilbao , pp.65-66
			self.waveform[i] = np.sum(A * np.exp(self.decay * i) * np.sin(i * omega)) / (omega.shape[0] * np.max(A))

	def getLabels(self) -> dict[str, list[Union[float, int]]]:
		'''
		Return the labels of the bessel model.
		'''

		return {'drum_size': [self.L], 'strike_location': [self.strike[0], self.strike[1]]} if hasattr(self, 'L') else {}

	def updateProperties(self, i: Union[int, None] = None) -> None:
		'''
		For every five drum samples generated, update the size of the drum. And for every drum sample generated update the
		strike location - the first strike location is always the centroid.
		'''

		if i is None or i % 5 == 0:
			# initialise a random drum size and strike location in the centroid of the drum.
			self.L = random.uniform(0.1, 2.)
			self.gamma = self.c / self.L
			self.strike = (0., 0.)
		else:
			# otherwise update the strike location to be a random location.
			self.strike = (
				random.uniform(-1., 1.),
				random.uniform(0., np.pi),
			)
