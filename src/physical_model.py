'''
'''

# core
import math
import os
import sys
from typing import Union

# dependencies
from numba import cuda			# GPU acceleration
import numpy as np				# maths
import numpy.typing as npt		# typing for numpy

# src
from audio_sampler import AudioSampler
from random_polygon import RandomPolygon
from settings import PhysicalModelSettings, settings
pm_settings: PhysicalModelSettings = settings['pm_settings']

# add the CUDA SDK to the environment variables
if settings['numba_path_2_cuda']:
	os.environ['CUDA_HOME'] = settings['numba_path_2_cuda']
	if not cuda.is_available():
		print(f'WARNING{"" if sys.platform not in ["linux", "darwin"] else "❗️"}')
		print('GPU support is not available for generating a physical model.')


class DrumModel(AudioSampler):
	'''
	This class creates a 2D simulation of an arbitrarily shaped drum, calculated
	using a FDTD scheme.
	'''

	# user-defined variables
	a: float					# maximum amplitude of the simulation ∈ [0, 1]
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
	s_0: float					# the first constant used in the FDTD
	s_1: float					# the second constant used in the FDTD
	d: float					# decay factor ∈ [0, 1]
	strike: tuple[int, int]		# where is the drum struck?
	# classes
	shape: RandomPolygon		# the shape of the drum

	def __init__(
		self,
		a: float = 1.0,
		allow_concave: bool = pm_settings['allow_concave'],
		decay_time: float = pm_settings['decay_time'],
		drum_size: float = pm_settings['drum_size'],
		material_density: float = pm_settings['material_density'],
		max_vertices: int = pm_settings['max_vertices'],
		tension: float = pm_settings['tension'],
	) -> None:
		'''
		Initialise the internal constants used for every simulation.
		params:
			a					Maximum amplitude of the simulation ∈ [0, 1].
			allow_concave		Can the drum be a concave shape?
			drum_size			Width and height of the simulation (m)
			max_vertices		What is the maximum number of vertices a given drum can have?
			material_density	Material density (kg/m^2)
			tension				Tension at rest (N/m)
		'''

		# initialise user defined variables
		self.a = a
		self.allow_concave = allow_concave
		self.L = drum_size
		self.max_vertices = max_vertices
		self.p = material_density
		self.t = tension
		self.t_60 = decay_time
		# initialise inferences
		self.k = 1 / self.sr
		self.c = (self.t / self.p) ** 0.5
		self.gamma = self.c / self.L
		self.H = math.floor((1 / (2 ** 0.5)) / (self.gamma * self.k))
		self.h = 1 / self.H
		self.cfl = self.gamma * self.k / self.h
		self.s_0 = self.cfl ** 2
		self.s_1 = 2 - 4 * self.cfl ** 2
		self.d = (1 - (6 * np.log(10) / self.t_60) * self.k) / (1 + (6 * np.log(10) / self.t_60) * self.k)

	def generateWaveform(self) -> None:
		'''
		Generate an audio waveform using a 2D FDTD scheme, as described by Stefan
		Bilbao in his book Numerical Sound Synthesis. Each scheme is initialised
		with a random shape and a unique strike location. The initial impulse for
		the model is a raised cosine distribution. Finally, the audio is generated
		using the main update loop.
		'''

		# types 🙇‍♂️
		u: npt.NDArray[np.float64]			# the FDTD grid
		u_0: npt.NDArray[np.float64]		# the FDTD grid at t = 0
		u_1: npt.NDArray[np.float64]		# the FDTD grid at t = 1
		x_range: tuple[int, int]			# range of the update equation across the x axis
		y_range: tuple[int, int]			# range of the update equation across the y axis

		u = np.zeros((self.H + 2, self.H + 2))
		u_0 = np.copy(u)
		u_1 = self.a * raisedCosine((self.H + 2, self.H + 2), self.strike)
		x_range = (
			round(np.min(self.shape.vertices[:, 0] * self.H)) + 1,
			round(np.max(self.shape.vertices[:, 0] * self.H)) + 1,
		)
		y_range = (
			round(np.min(self.shape.vertices[:, 1] * self.H)) + 1,
			round(np.max(self.shape.vertices[:, 1] * self.H)) + 1,
		)

		# FDTD w/ GPU
		if cuda.is_available():
			threads_per_block = (16, 16)
			blocks_per_grid = (
				math.ceil((x_range[1] - x_range[0]) / threads_per_block[0]),
				math.ceil((y_range[1] - y_range[0]) / threads_per_block[1]),
			)

			for i in range(self.length):
				# handle initial events
				if i == 0:
					self.waveform[i] = 0.0
					continue
				if i == 1:
					self.waveform[i] = u_1[self.strike]
					continue

				# main loop
				if i % 2 == 0:
					self.FDTDKernal[blocks_per_grid, threads_per_block](
						u,
						u_1,
						u_0,
						self.shape.mask,
						x_range[0],
						y_range[0],
						self.s_0,
						self.s_1,
						self.d,
					)
					u_0 = np.copy(u)

				if i % 2 == 1:
					self.FDTDKernal[blocks_per_grid, threads_per_block](
						u,
						u_0,
						u_1,
						self.shape.mask,
						x_range[0],
						y_range[0],
						self.s_0,
						self.s_1,
						self.d,
					)
					u_1 = np.copy(u)
				self.waveform[i] = u[self.strike]
				self._sample_count += 1

		# FDTD w/o GPU
		else:
			for i in range(self.length):
				# handle initial events
				if i == 0:
					self.waveform[i] = 0.0
					continue
				if i == 1:
					self.waveform[i] = u_1[self.strike]
					continue

				# main loop
				if i % 2 == 0:
					for x in range(*x_range):
						for y in range(*y_range):
							# dirichlet  boundary condition
							if self.shape.mask[x - 1, y - 1] == 0:
								continue
							u[x, y] = (self.s_0 * sum([
								u_1[x, y + 1],
								u_1[x + 1, y],
								u_1[x, y - 1],
								u_1[x - 1, y],
							])) + (self.s_1 * u_0[x, y]) - (self.d * u_0[x, y])
					u_0 = np.copy(u)

				if i % 2 == 1:
					for x in range(*x_range):
						for y in range(*y_range):
							# dirichlet  boundary condition
							if self.shape.mask[x - 1, y - 1] == 0:
								continue
							u[x, y] = (self.s_0 * sum([
								u_0[x, y + 1],
								u_0[x + 1, y],
								u_0[x, y - 1],
								u_0[x - 1, y],
							])) + (self.s_1 * u_0[x, y]) - (self.d * u_1[x, y])
					u_1 = np.copy(u)

				self.waveform[i] = u[self.strike]
				self._sample_count += 1

	def getLabels(self) -> list[Union[float, int]]:
		'''
		Return the labels for the currently generated audio sample.
		'''

		if hasattr(self, 'shape'):
			out = np.zeros((self.max_vertices, 2))
			for i in range(len(self.shape.vertices)):
				out[i] = self.shape.vertices[i]
			return out.tolist()
		else:
			return []

	def updateProperties(self) -> None:

		if self._sample_count % 5 == 0:
			# initialise a random drum shape and calculate the initial conditions
			# relative to the centroid of the drum.
			self.shape = RandomPolygon(
				self.max_vertices,
				grid_size=self.H,
				allow_concave=self.allow_concave,
			)
			self.strike = (
				round(self.shape.centroid[0] * self.H),
				round(self.shape.centroid[1] * self.H),
			)
		else:
			# random strike
			self.strike = (0, 0)
			while not self.shape.mask[self.strike]:
				self.strike = (np.random.randint(0, self.H), np.random.randint(0, self.H))

	@staticmethod
	@cuda.jit
	def FDTDKernal(
		u: npt.NDArray[np.float64],
		u_minus_1: npt.NDArray[np.float64],
		u_minus_2: npt.NDArray[np.float64],
		mask: npt.NDArray[np.int8],
		x_0: int,
		y_0: int,
		s_0: float,
		s_1: float,
		d: float,
	) -> None:
		x, y = cuda.grid(2)
		x += x_0
		y += y_0
		if mask[x - 1, y - 1] != 0 and x < u.shape[0] - 2 and y < u.shape[1] - 2:
			u[x, y] = (s_0 * (
				u_minus_1[x, y + 1]
				+ u_minus_1[x + 1, y]
				+ u_minus_1[x, y - 1]
				+ u_minus_1[x - 1, y]
			)) + (s_1 * u_minus_1[x, y]) - (d * u_minus_2[x, y])


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
