# core
import math
import os

# dependencies
from numba import cuda			# GPU acceleration
import numpy as np				# maths
import numpy.typing as npt		# typing for numpy

# src
from audio_sample import AudioSample
from geometry import RandomPolygon
from settings import PhysicalModelSettings, settings
pmSettings: PhysicalModelSettings = settings['PM_SETTINGS']


# add the CUDA SDK to the environment variables
if pmSettings['path_2_cuda'] is not None:
	os.environ['CUDA_HOME'] = pmSettings['path_2_cuda']
	if not cuda.is_available():
		print('WARNING❗️ Nvidia GPU support is not available for generating a physical model.')


class PhysicalModel(AudioSample):
	'''
	'''
	# variables
	L: float = 0.3 # roughly 12"			# width and height of the simulation (m)
	p: float = 0.26							# material density (kg/m^2)
	t: float = 2000							# tension at rest (N/m)

	# inferrences
	k: float = 1 / settings['SAMPLE_RATE']	# sample length (ms)
	c: float = (t / p) ** 0.5				# wavespeed (m/s)
	gamma: float = c / L					# scaled wavespeed (1/s)
	H: float = math.floor(1 / (gamma * k))	# number of grid points across each dimension, for the domain U ∈ [0, 1]
	h: float = 1 / H						# length of each grid step
	cfl: float = gamma * k / h				# courant number

	def __init__(self) -> None:
		self.shape = RandomPolygon(self.H)
		super().__init__(y=self.shape.vertices.tolist())

	def generateWaveform(self) -> npt.NDArray[np.float64]:
		waveform = np.zeros(self.duration)
		return waveform
