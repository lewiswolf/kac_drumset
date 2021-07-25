# core
import os

# dependencies
from numba import cuda			# GPU acceleration
import numpy as np				# maths
import numpy.typing as npt		# typing for numpy

# src
from settings import settings
from audio_sample import AudioSample


# add the CUDA SDK to the environment variables
if settings['PATH_2_CUDA'] is not None:
	if 'CUDA_HOME' not in os.environ:
		os.environ['CUDA_HOME'] = settings['PATH_2_CUDA']
	if not cuda.is_available():
		print('WARNING❗️ Nvidia GPU support is not available.')


class PhysicalModel(AudioSample):
	def init(self) -> None:
		self.y = []

	def generateWaveform(self) -> npt.NDArray[np.float64]:
		return np.zeros(0)
