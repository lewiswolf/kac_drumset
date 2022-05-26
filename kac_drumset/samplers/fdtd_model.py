'''
'''

# core
import math
from typing import Union

# dependencies
# from numba import cuda	# GPU acceleration
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from ..dataset import AudioSampler, SamplerSettings
from ..geometry import RandomPolygon, booleanMask
# from ..physics import raisedCosine


class FDTDModel(AudioSampler):
	'''
	This class creates a 2D simulation of an arbitrarily shaped drum, calculated
	using a FDTD scheme.
	'''

	# user-defined variables
	a: float					# maximum amplitude of the simulation ∈ [0, 1]
	L: float					# size of the drum, spanning both the horizontal and vertical axes (m)
	max_vertices: int			# maximum amount of vertices for a given drum
	p: float					# material density of the simulated drum membrane (kg/m^2)
	t: float					# tension at rest (N/m)
	# FDTD inferences
	k: float					# sample length (ms)
	c: float					# wavespeed (m/s)
	gamma: float				# scaled wavespeed (1/s)
	H: int						# number of grid points across each dimension, for the domain U ∈ [0, 1]
	h: float					# length of each grid step
	cfl: float					# courant number
	s_0: float					# the first constant used in the FDTD
	s_1: float					# the second constant used in the FDTD
	d: float					# decay factor ∈ [0, 1]
	# drum properties
	B: npt.NDArray[np.int8]		# boolean matrix define the boundary conditions for the drum
	shape: RandomPolygon		# the shape of the drum
	strike: tuple[int, int]		# where is the drum struck?

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

	def __init__(
		self,
		duration: float,
		sample_rate: int,
		amplitude: float = 1.0,
		decay_time: float = 1.0,
		drum_size: float = 0.3,
		material_density: float = 0.26,
		max_vertices: int = 10,
		tension: float = 2000.0,
	) -> None:
		''' Initialise sampler. '''

		super().__init__(duration, sample_rate)
		# initialise user defined variables
		self.a = amplitude
		self.L = drum_size
		self.max_vertices = max_vertices
		self.p = material_density
		self.t = tension
		self.t_60 = decay_time
		# initialise inferences
		self.k = 1 / self.sample_rate
		self.c = (self.t / self.p) ** 0.5
		self.gamma = self.c / self.L
		self.H = math.floor((1 / (2 ** 0.5)) / (self.gamma * self.k))
		self.h = 1 / self.H
		self.cfl = self.gamma * self.k / self.h
		self.s_0 = self.cfl ** 2
		self.s_1 = 2 - 4 * self.cfl ** 2
		self.d = (1 - (6 * np.log(10) / self.t_60) * self.k) / (1 + (6 * np.log(10) / self.t_60) * self.k)

	def generateWaveform(self) -> None:
		pass

	def getLabels(self) -> list[Union[float, int]]:
		''' This method should return the y labels for the generated audio. '''

		if hasattr(self, 'shape'):
			return np.pad(
				self.shape.vertices,
				[(0, self.max_vertices - self.shape.n), (0, 0)],
				mode='constant',
			).tolist()
		else:
			return []

	def updateProperties(self, i: Union[int, None] = None) -> None:
		'''
		For every five drum samples generated, update the drum shape. And for every drum sample generated update the strike
		location - the first strike location is always the centroid.
		'''

		if i is None or i % 5 == 0:
			# initialise a random drum shape and calculate the initial conditions relative to the centroid of the drum.
			self.shape = RandomPolygon(self.max_vertices)
			self.B = booleanMask(self.shape.vertices, self.H, self.shape.convex)
			self.strike = (
				round(self.shape.centroid[0] * self.H),
				round(self.shape.centroid[1] * self.H),
			)
		else:
			# otherwise update the strike location to be a random location.
			self.strike = (0, 0)
			while not self.B[self.strike]:
				self.strike = (np.random.randint(0, self.H), np.random.randint(0, self.H))
