# core
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
		print('WARNING❗️ Nvidia GPU support is not available.')


class PhysicalModel(AudioSample):
	'''
	'''

	def init(self) -> None:
		self.shape = RandomPolygon()
		self.y = []

	def generateWaveform(self) -> npt.NDArray[np.float64]:
		return np.zeros(0)
