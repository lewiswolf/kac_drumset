'''
This sampler is used to produce physically modelled, arbitrarily shaped drums. This is achieved using a randomly
generated polygon, which is used to define the boundary conditions, and a finite difference time domain simulation.
'''

# core
import math

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from kac_prediction.dataset import classLocalsToKwargs, AudioSampler, SamplerSettings
from ..physics import FDTDWaveform1D, raisedCosine

__all__ = [
	'LinearFDTD',
]


class LinearFDTD(AudioSampler):
	'''
	This class creates a 1D simulation, calculated using a FDTD scheme.
	'''

	# user-defined variables
	a: float						# maximum amplitude of the simulation ∈ [0, 1]
	d_60: float						# decay time (seconds)
	p: float						# material density of the simulated drum membrane (kg/m^2)
	strike_width: float				# width of the drum strike (m)
	t: float						# tension at rest (N/m)
	# FDTD inferences
	L: float						# size of the drum, spanning both the horizontal and vertical axes (m)
	c: float						# wavespeed (m/s)
	cfl: float						# courant number
	gamma: float					# scaled wavespeed (1/s)
	H: int							# number of grid points across each dimension, for the domain U ∈ [0, 1]
	h: float						# length of each grid step
	k: float						# sample length (ms)
	sigma: float					# half strike width relative to L
	sigma_2: float					# sigma ** 2 relative to H
	# FDTD update coefficients
	c_0: float						# first coefficient
	c_1: float						# second coefficient
	c_2: float						# third coefficient
	u_0: npt.NDArray[np.float64]	# initial conditions for each simulation
	# drum properties
	strike: float					# where is the drum struck?
	w: float						# sample point of the 2D surface

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

	def __init__(
		self,
		duration: float,
		sample_rate: int,
		amplitude: float = 1.,
		decay_time: float = 2.,
		material_density: float = 0.2,
		strike_width: float = 0.02,
		tension: float = 2000.,
	) -> None:
		'''
		When the class is first instantiated, all of its physical properties are inferred from the user parameters.
		'''

		# initialise settings
		super().__init__(**classLocalsToKwargs(locals()))
		# initialise user defined variables
		self.a = amplitude
		self.d_60 = decay_time
		self.p = material_density
		self.strike_width = strike_width
		self.t = tension
		self.w = 0.5
		# initialise inferences
		self.k = 1. / self.sample_rate
		self.c = (self.t / self.p) ** 0.5

	def __updateCoefficients__(self) -> None:
		self.gamma = self.c / self.L
		self.H = math.floor(1. / (self.gamma * self.k))
		self.h = 1. / self.H
		self.cfl = self.gamma * self.k / self.h
		self.sigma = self.strike_width * 0.5 / self.L
		self.sigma_2 = max((self.sigma * self.H) ** 2., 1.)
		# FDTD update coefficients
		log_decay = self.k * 6. * np.log(10.) / self.d_60
		self.c_0 = (self.cfl ** 2.) / (1. + log_decay)
		self.c_1 = (2. - 2. * (self.cfl ** 2.)) / (1. + log_decay)
		self.c_2 = (1. - log_decay) / (1. + log_decay)
		self.u_0 = np.zeros((self.H + 2, ))

	def generateWaveform(self) -> None:
		''' Calculate the FDTD for a 1D simulation. '''

		if hasattr(self, 'L'):
			self.waveform = FDTDWaveform1D(
				self.u_0,
				np.pad(
					self.a * raisedCosine((self.strike, ), (self.H, ), sigma=self.sigma) / self.sigma_2,
					1,
					mode='constant',
				),
				self.c_0,
				self.c_1,
				self.c_2,
				self.length,
				self.w,
			)

	def getLabels(self) -> dict[str, list[float | int]]:
		''' This method returns the labels for the FDTD. '''
		return {'size': [self.L], 'strike_location': [self.strike]} if hasattr(self, 'L') else {}

	def updateProperties(self, i: int | None = None) -> None:
		'''
		For every five drum samples generated, update the size of the drum. And for every drum sample generated update the
		strike location - the first strike location is always the centroid.
		'''

		if i is None or i % 5 == 0:
			# initialise a random drum size and strike location in the centroid of the drum.
			self.L = np.random.uniform(0.1, 1.)
			self.__updateCoefficients__()
			self.strike = 0.5
		else:
			# otherwise update the strike location to be a random location.
			self.strike = np.random.uniform(0., 1.)
