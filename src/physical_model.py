# core
import math
import os
from typing import Union

# dependencies
from numba import cuda			# GPU acceleration
import numpy as np				# maths
import numpy.typing as npt		# typing for numpy

# src
from audio_sampler import AudioSampler
from geometry import RandomPolygon
from settings import PhysicalModelSettings, settings
pm_settings: PhysicalModelSettings = settings['pm_settings']

# add the CUDA SDK to the environment variables
if pm_settings['path_2_cuda']:
	os.environ['CUDA_HOME'] = pm_settings['path_2_cuda']
	if not cuda.is_available():
		print('WARNING❗️ Nvidia GPU support is not available for generating a physical model.')


class DrumModel(AudioSampler):
	'''
	This class creates a 2D simulation of an arbitrarily shaped drum, calculated
	using a FDTD scheme.
	'''

	# user-defined variables
	a: float					# maximum amplitude of the simulation ∈ [0, 1].
	allow_concave: bool			# can the drum be a concave shape?
	L: float					# width and height of the simulation (m)
	max_vertices: int			# what is the maximum number of vertices a given drum can have?
	p: float					# material density (kg/m^2)
	t: float					# tension at rest (N/m)
	# inferrences
	k: float					# sample length (ms)
	c: float					# wavespeed (m/s)
	gamma: float				# scaled wavespeed (1/s)
	H: int						# number of grid points across each dimension, for the domain U ∈ [0, 1]
	h: float					# length of each grid step
	cfl: float					# courant number
	# classes
	shape: RandomPolygon		# the shape of the drum

	def __init__(
		self,
		a: float = 1.0,
		allow_concave: bool = pm_settings['allow_concave'],
		L: float = pm_settings['drum_size'],
		max_vertices: int = pm_settings['max_vertices'],
		p: float = pm_settings['material_density'],
		t: float = pm_settings['tension'],
	) -> None:
		'''
		Initialise the internal constants used for every simulation.
		params:
			a				Maximum amplitude of the simulation ∈ [0, 1].
			allow_concave	Can the drum be a concave shape?
			L				Width and height of the simulation (m)
			max_vertices	What is the maximum number of vertices a given drum can have?
			p				Material density (kg/m^2)
			t				Tension at rest (N/m)
		'''

		# initialise user defined variables
		self.a = a
		self.allow_concave = allow_concave
		self.L = L
		self.max_vertices = max_vertices
		self.p = p
		self.t = t
		# initialise inferences
		self.k = 1 / self.sr
		self.c = (self.t / self.p) ** 0.5
		self.gamma = self.c / self.L
		self.H = math.floor((1 / (2 ** 0.5)) / (self.gamma * self.k))
		self.h = 1 / self.H
		self.cfl = self.gamma * self.k / self.h

	def generateWaveform(self) -> None:
		'''
		Generate an audio waveform using a 2D FDTD scheme, as described by Stefan
		Bilbao in his book Numerical Sound Synthesis. Each scheme is initialised
		with a random shape and a unique strike location. The initial impulse for
		the model is a raised cosine distribution. Finally, the audio is generated
		using the main update loop.
		'''

		s_0: float = self.cfl ** 2			# the first constant in the FDTD update equation
		s_1: float = 2 - 4 * self.cfl ** 2	# the second constant in the FDTD update equation
		strike: tuple[int, int]				# strike location
		u: npt.NDArray[np.float64]			# the FDTD grid
		u_0: npt.NDArray[np.float64]		# the FDTD grid at t = 0
		u_1: npt.NDArray[np.float64]		# the FDTD grid at t = 1
		x_range: tuple[int, int]			# range of the update equation across the x axis
		y_range: tuple[int, int]			# range of the update equation across the y axis

		# initialise a random drum shape
		self.shape = RandomPolygon(
			self.max_vertices,
			grid_size=self.H,
			allow_concave=self.allow_concave,
		)
		strike = (
			round(self.shape.centroid[0] * self.H),
			round(self.shape.centroid[1] * self.H),
		)

		# calculate the initial conditions relative to the centroid of the drum
		u = np.zeros((self.H + 2, self.H + 2))
		u_0 = np.copy(u)
		u_1 = self.a * raisedCosine((self.H + 2, self.H + 2), strike)
		x_range = (
			round(np.min(self.shape.vertices[:, 0] * self.H)) + 1,
			round(np.max(self.shape.vertices[:, 0] * self.H)) + 1,
		)
		y_range = (
			round(np.min(self.shape.vertices[:, 1] * self.H)) + 1,
			round(np.max(self.shape.vertices[:, 1] * self.H)) + 1,
		)

		for i in range(self.length):
			# handle initial events
			if i == 0:
				self.waveform[i] = 0.0
				continue
			if i == 1:
				self.waveform[i] = u_1[strike]
				continue

			# main loop
			if i % 2 == 0:
				for x in range(*x_range):
					for y in range(*y_range):
						# dirichlet  boundary condition
						if self.shape.mask[x - 1, y - 1] == 0:
							continue
						u[x, y] = (s_0 * sum([
							u_1[x, y + 1],
							u_1[x + 1, y],
							u_1[x, y - 1],
							u_1[x - 1, y],
						])) + (s_1 * u_1[x, y]) - u_0[x, y]
				u_0 = np.copy(u)

			if i % 2 == 1:
				for x in range(*x_range):
					for y in range(*y_range):
						# dirichlet  boundary condition
						if self.shape.mask[x - 1, y - 1] == 0:
							continue
						u[x, y] = (s_0 * sum([
							u_0[x, y + 1],
							u_0[x + 1, y],
							u_0[x, y - 1],
							u_0[x - 1, y],
						])) + (s_1 * u_0[x, y]) - u_1[x, y]
				u_1 = np.copy(u)

			self.waveform[i] = u[strike]

	def getLabels(self) -> list[Union[float, int]]:
		'''
		Return the labels for the currently generated audio sample.
		'''

		if hasattr(self, 'shape'):
			return self.shape.vertices.tolist()
		else:
			return []


def raisedCosine(
	H: tuple[int, ...],
	contact_point: tuple[int, ...],
	sigma: float = 0.5,
) -> npt.NDArray[np.float64]:
	'''
	This functions creates a raised cosine distribution of size H, centered at
	the contact_point. Only 1D and 2D distributions are supported.
	params:
			H				A tuple representing the size of the output matrix.
			contact_point	The coordinate used to represent the centre of the
											cosine distribution.
			sigma			The radius of the distribution.
	'''

	# handle dimensions > 2 and incompatible inputs
	if len(contact_point) > 2 or len(H) > 2 or len(H) != len(contact_point):  # gross
		raise ValueError()

	# solve for the one dimensional case
	if len(contact_point) == 1:
		rc = np.zeros(H)
		for x in range(H[0]):
			x_diff = x - contact_point[0]
			if abs(x_diff) <= sigma:
				rc[x] = 0.5 * (1 + np.cos(np.pi * x_diff / sigma))

	# solve for the two dimensional case
	if len(contact_point) == 2:
		rc = np.zeros(H)
		for x in range(H[0]):
			for y in range(H[1]):
				l2_norm = (((x - contact_point[0]) ** 2) + ((y - contact_point[1]) ** 2)) ** 0.5
				if l2_norm <= sigma:
					rc[x, y] = 0.5 * (1 + np.cos(np.pi * l2_norm / sigma))

	return rc
