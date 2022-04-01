'''
Utility functions for use whilst developing, as well as part of unit_tests.py.
'''

# core
from typing import Union

# dependencies
import numpy as np 			# maths

# src
from .audio_sampler import AudioSampler


class TestSweep(AudioSampler):
	'''
	This class produces a sine wave sweep across the audio spectrum, from 20hz to f_s / 2.
	'''

	def __init__(self, duration: float = 1.0, sr: int = 44100) -> None:
		'''
		Render the sinusoidal sweep.
		'''

		super().__init__(duration, sr)
		phi = 0.0
		s_l = 1 / self.sr
		two_pi = 2 * np.pi

		for t in range(self.length):
			f = 20 + ((t / self.length) ** 2) * ((self.sr / 2) - 20)
			self.waveform[t] = np.sin(phi)
			phi += two_pi * f * s_l
			phi -= two_pi if phi > two_pi else 0

	def generateWaveform(self) -> None:
		pass

	def getLabels(self) -> list[Union[float, int]]:
		return []

	def updateProperties(self) -> None:
		pass
