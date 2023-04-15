'''
An AudioSampler() used for generating a sinusoidal sweep of the audio spectrum.
'''

# dependencies
import numpy as np 			# maths

# src
from ..dataset import AudioSampler, SamplerSettings
from ..dataset.utils import classLocalsToKwargs

__all__ = [
	'TestSweep',
]


class TestSweep(AudioSampler):
	'''
	This class produces a sine wave sweep across the audio spectrum, from 20hz to f_s / 2.
	'''

	class Settings(SamplerSettings):
		pass

	def __init__(self, duration: float, sample_rate: int) -> None:
		'''
		Render the sinusoidal sweep.
		'''

		super().__init__(**classLocalsToKwargs(locals()))
		phi = 0.
		s_l = 1 / sample_rate
		two_pi = 2 * np.pi

		for t in range(self.length):
			f = 20 + ((t / self.length) ** 2) * ((sample_rate / 2) - 20)
			self.waveform[t] = np.sin(phi)
			phi += two_pi * f * s_l
			phi -= two_pi if phi > two_pi else 0

	def generateWaveform(self) -> None:
		pass

	def getLabels(self) -> dict[str, list[float | int]]:
		return {}

	def updateProperties(self, i: int | None = None) -> None:
		pass
