'''
An AudioSampler() used for generating a random waveform.
'''

# core
import random
from typing import Literal, TypeAlias

# dependencies
import numpy as np 	# maths

# src
from ..dataset import AudioSampler, SamplerSettings
from ..dataset.utils import classLocalsToKwargs

__all__ = [
	'TestTone',
]


Waveshape: TypeAlias = Literal['saw', 'sin', 'sqr', 'tri']


class TestTone(AudioSampler):
	'''
	This class produces an arbitrary test tone, using either a sawtooth, sine, square or triangle waveform. If it's
	initial frequency is not set, it will automatically create random frequencies.
	'''

	f_0: float					# fundamental frequency (hz)
	__random_f_0: bool			# is the fundamental random or fixed?
	waveshape: Waveshape		# shape of the waveform

	class Settings(SamplerSettings, total=False):
		f_0: float				# fixed fundamental frequency
		waveshape: Waveshape	# shape of the waveform

	def __init__(
		self,
		duration: float,
		sample_rate: int,
		f_0: float = 0.,
		waveshape: Waveshape = 'sin',
	) -> None:
		'''
		Initialise a random waveform generator.
		input:
			f_0 		Fundamental frequency (hz).
			waveshape 	Shape of the waveform.
		'''

		super().__init__(**classLocalsToKwargs(locals()))
		self.f_0 = f_0
		self.__random_f_0 = not bool(f_0)
		self.waveshape = waveshape

	def generateWaveform(self) -> None:
		'''
		Renders a specified waveform to a numpy array.
		'''

		f_t = self.f_0 * np.arange(self.length) / self.sample_rate
		if self.waveshape == 'saw':
			self.waveform = 2. * np.array([i % 1 for i in f_t]) - 1.
		if self.waveshape == 'sin':
			self.waveform = np.sin(2 * np.pi * f_t)
		if self.waveshape == 'sqr':
			self.waveform = np.array([-0.95 if i < 0 else 0.95 for i in np.sin(2 * np.pi * f_t)])
		if self.waveshape == 'tri':
			self.waveform = 4. * np.array([1 - (i % 1) if i % 1 > 0.5 else i % 1 for i in f_t]) - 1.

	def getLabels(self) -> dict[str, list[float | int]]:
		''' Returns f_0 as a label. '''
		return {'f_0': [self.f_0]} if self.f_0 else {}

	def updateProperties(self, i: int | None = None) -> None:
		''' Randomise f_0. '''
		if self.__random_f_0:
			self.f_0 = random.uniform(110, 880)
