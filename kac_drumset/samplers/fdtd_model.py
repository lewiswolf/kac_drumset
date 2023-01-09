'''
This sampler is used to produce physically modelled, arbitrarily shaped drums. This is achieved using a randomly
generated polygon, which is used to define the boundary conditions, and a finite difference time domain simulation.
'''

# core
import math
from typing import Union

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from ..dataset import AudioSampler, SamplerSettings
from ..dataset.utils import classLocalsToKwargs
from ..geometry import RandomPolygon, booleanMask, isPointInsidePolygon
from ..physics import FDTDWaveform2D, raisedCosine

__all__ = [
	'FDTDModel',
]


class FDTDModel(AudioSampler):
	'''
	This class creates a 2D simulation of an arbitrarily shaped drum, calculated using a FDTD scheme.
	'''

	# user-defined variables
	a: float					# maximum amplitude of the simulation ∈ [0, 1]
	d_60: float					# decay time (seconds)
	L: float					# size of the drum, spanning both the horizontal and vertical axes (m)
	max_vertices: int			# maximum amount of vertices for a given drum
	p: float					# material density of the simulated drum membrane (kg/m^2)
	strike_width: float			# width of the drum strike (m)
	t: float					# tension at rest (N/m)
	# FDTD inferences
	c: float					# wavespeed (m/s)
	cfl: float					# courant number
	gamma: float				# scaled wavespeed (1/s)
	H: int						# number of grid points across each dimension, for the domain U ∈ [0, 1]
	h: float					# length of each grid step
	k: float					# sample length (ms)
	sigma: float				# strike width relative to H
	sigma_2: float				# sigma ** 2
	# FDTD update coefficients
	c_0: float					# first coefficient
	c_1: float					# second coefficient
	c_2: float					# third coefficient
	# drum properties
	B: npt.NDArray[np.int8]		# boolean matrix define the boundary conditions for the drum
	shape: RandomPolygon		# the shape of the drum
	strike: tuple[float, float]	# where is the drum struck?

	class Settings(SamplerSettings, total=False):
		'''
		This is an abstract TypedDict used to mirror the type declaration for the customised __init__() method. This allows
		for type safety when using a custom AudioSampler with an arbitrary __init__() method.
		'''

		amplitude: float			# maximum amplitude of the simulation ∈ [0, 1]
		decay_time: float			# how long will the simulation take to decay? (seconds)
		drum_size: float			# size of the drum, spanning both the horizontal and vertical axes (m)
		material_density: float		# material density of the simulated drum membrane (kg/m^2)
		max_vertices: int			# maximum amount of vertices for a given drum
		strike_width: float			# size of the strike location (cm)
		tension: float				# tension at rest (N/m)

	def __init__(
		self,
		duration: float,
		sample_rate: int,
		amplitude: float = 1.,
		decay_time: float = 2.,
		drum_size: float = 0.3,
		material_density: float = 0.2,
		max_vertices: int = 10,
		strike_width: float = 0.005,
		tension: float = 2000.,
	) -> None:
		'''
		When the class is first instantiated, all of its physical properties are inferred from the user parameters.
		'''

		# initialise user defined variables
		super().__init__(**classLocalsToKwargs(locals()))
		self.a = amplitude
		self.d_60 = decay_time
		self.L = drum_size
		self.max_vertices = max_vertices
		self.p = material_density
		self.strike_width = strike_width
		self.t = tension
		# initialise inferences
		self.k = 1 / self.sample_rate
		self.c = (self.t / self.p) ** 0.5
		self.gamma = self.c / self.L
		self.H = math.floor((1 / (2 ** 0.5)) / (self.gamma * self.k))
		self.h = 1 / self.H
		self.cfl = self.gamma * self.k / self.h
		self.sigma = self.H * strike_width / self.L
		self.sigma_2 = self.sigma ** 2.
		# FDTD update coefficients
		log_decay = self.k * 6 * np.log(10) / self.d_60
		self.c_0 = (self.cfl ** 2) / (1 + log_decay)
		self.c_1 = (2 - 4 * (self.cfl ** 2)) / (1 + log_decay)
		self.c_2 = (1 - log_decay) / (1 + log_decay)

	def generateWaveform(self) -> None:
		'''
		Calculate the FDTD for a 2D polygon.
		'''

		if hasattr(self, 'shape'):
			self.waveform = FDTDWaveform2D(
				np.zeros((self.H + 2, self.H + 2)),
				np.pad(self.a * raisedCosine(
					(self.H, self.H),
					(self.strike[0] * (self.H - 1), self.strike[1] * (self.H - 1)),
					sigma=self.sigma,
				) / self.sigma_2, 1, mode='constant'),
				np.pad(self.B, 1, mode='constant'),
				self.c_0,
				self.c_1,
				self.c_2,
				self.length,
				(
					round(self.shape.centroid[0] * (self.H - 1)) + 1,
					round(self.shape.centroid[1] * (self.H - 1)) + 1,
				),
			)

	def getLabels(self) -> dict[str, list[Union[float, int]]]:
		'''
		This method returns the vertices for the drum's arbitrary shaped, padded with zeros to equal the length of
		self.max_vertices.
		'''

		return {
			'strike_location': [*self.strike],
			'vertices': self.shape.vertices.tolist(),
		} if hasattr(self, 'shape') else {}

	def updateProperties(self, i: Union[int, None] = None) -> None:
		'''
		For every five drum samples generated, update the drum shape. And for every drum sample generated update the strike
		location - the first strike location is always the centroid.
		'''

		if i is None or i % 5 == 0:
			# initialise a random drum shape and calculate the initial conditions relative to the centroid of the drum.
			self.shape = RandomPolygon(self.max_vertices)
			self.B = booleanMask(self.shape, self.H)
			self.strike = self.shape.centroid
		else:
			# otherwise update the strike location to be a random location.
			self.strike = (np.random.uniform(0., 1.), np.random.uniform(0., 1.))
			while not isPointInsidePolygon(self.strike, self.shape):
				self.strike = (np.random.uniform(0., 1.), np.random.uniform(0., 1.))
