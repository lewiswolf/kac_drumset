'''
Utility functions used whilst developing, as well as alongside unit_tests.py.
'''

# core
import cProfile
import pstats
import random
from typing import Any, Callable, Literal

# dependencies
import numpy as np						# maths
import numpy.typing as npt				# typing for numpy

# src
from audio_sample import AudioSample


class TestSweep(AudioSample):
	'''
	This class produces a sine wave sweep across the audio spectrum, from 20hz
	to f_s / 2.
	'''

	def __init__(self) -> None:
		super().__init__()

	def generateWaveform(self) -> npt.NDArray[np.float64]:
		phi: float = 0.0
		s_l: float = 1 / self.sr
		two_pi: float = 2 * np.pi
		waveform: npt.NDArray[np.float64] = np.zeros((self.length))

		for i in range(self.length):
			f: float = 20 + ((i / self.length) ** 2) * ((self.sr / 2) - 20)
			waveform[i] = np.sin(phi)
			phi += two_pi * f * s_l
			if phi > two_pi:
				phi -= two_pi
		return waveform


class TestTone(AudioSample):
	'''
	This class produces an arbitrary test tone, such as a sine wave.
	'''

	def __init__(
		self,
		f0: float = 0.0,
		waveshape: Literal['saw', 'sin', 'sqr', 'tri'] = 'sin',
	) -> None:
		'''
		Initialise a random waveform generator.
		params:
			f0 			Fundamental frequency (hz).
			waveshape 	Shape of the waveform.
		'''

		self.f0 = f0 if f0 else random.uniform(110, 880)
		self.waveshape = waveshape
		super().__init__(y=[self.f0])

	def generateWaveform(self) -> npt.NDArray[np.float64]:
		'''
		Renders a specified waveform to a numpy array.
		'''

		f_t = self.f0 * (np.arange(self.length) / self.sr)
		if self.waveshape == 'saw':
			return 2.0 * np.array([i % 1 for i in f_t]) - 1.0
		if self.waveshape == 'sin':
			return np.sin(2 * np.pi * f_t)
		if self.waveshape == 'sqr':
			return np.array([-0.95 if i < 0 else 0.95 for i in np.sin(2 * np.pi * f_t)])
		if self.waveshape == 'tri':
			return 4.0 * np.array([1 - (i % 1) if i % 1 > 0.5 else i % 1 for i in f_t]) - 1.0


def withProfiler(func: Callable, n: int, *args: Any, **kwargs: Any) -> None:
	'''
	Calls the input function using cProfile to generate a performance report in the console.
	Prints the n most costly functions.
	'''

	with cProfile.Profile() as pr:
		func(*args, **kwargs)
	stats = pstats.Stats(pr)
	stats.sort_stats(pstats.SortKey.TIME)
	stats.print_stats(n)
