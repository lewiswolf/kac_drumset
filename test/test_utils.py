'''
Utility functions for use whilst developing, as well as part of unit_tests.py.
'''

# core
import contextlib
import cProfile
import pstats
import os
import random
import sys
from typing import Any, Callable, Iterator, Literal, Union

# dependencies
import numpy as np						# maths

# src
from audio_sampler import AudioSampler


class TestSweep(AudioSampler):
	'''
	This class produces a sine wave sweep across the audio spectrum, from 20hz
	to f_s / 2.
	'''

	def __init__(self) -> None:
		'''
		Render the sinusoidal sweep.
		'''
		phi: float = 0.0
		s_l: float = 1 / self.sr
		two_pi: float = 2 * np.pi

		for i in range(self.length):
			f: float = 20 + ((i / self.length) ** 2) * ((self.sr / 2) - 20)
			self.waveform[i] = np.sin(phi)
			phi += two_pi * f * s_l
			if phi > two_pi:
				phi -= two_pi

	def generateWaveform(self) -> None:
		pass

	def getLabels(self) -> list[Union[float, int]]:
		return []

	def updateProperties(self) -> None:
		pass


class TestTone(AudioSampler):
	'''
	This class produces an arbitrary test tone, such as a sine wave.
	'''

	f0: float										# fundamental frequency (hz)
	__random_f0: bool								# is the fundamental random or fixed?
	waveshape: Literal['saw', 'sin', 'sqr', 'tri']	# shape of the waveform

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

		self.f0 = f0
		self.__random_f0 = not bool(f0)
		self.waveshape = waveshape

	def generateWaveform(self) -> None:
		'''
		Renders a specified waveform to a numpy array.
		'''

		f_t = self.f0 * (np.arange(self.length) / self.sr)
		if self.waveshape == 'saw':
			self.waveform = 2.0 * np.array([i % 1 for i in f_t]) - 1.0
		if self.waveshape == 'sin':
			self.waveform = np.sin(2 * np.pi * f_t)
		if self.waveshape == 'sqr':
			self.waveform = np.array([-0.95 if i < 0 else 0.95 for i in np.sin(2 * np.pi * f_t)])
		if self.waveshape == 'tri':
			self.waveform = 4.0 * np.array([1 - (i % 1) if i % 1 > 0.5 else i % 1 for i in f_t]) - 1.0

	def getLabels(self) -> list[Union[float, int]]:
		''' Returns f0 as a label. '''
		return [self.f0] if self.f0 else []

	def updateProperties(self) -> None:
		''' Randomise f0. '''
		if self.__random_f0:
			self.f0 = random.uniform(110, 880)


@contextlib.contextmanager
def noPrinting(allow_errors: bool = False) -> Iterator[Any]:
	'''
	This wrapper can used around blocks of code to silece calls to print(), as well as
	optionally silence error messages.
	'''

	with open(os.devnull, 'w') as dummy_file:
		if not allow_errors:
			sys.stderr = dummy_file
		sys.stdout = dummy_file
		yield
		dummy_file.close()
	sys.stderr = sys.__stderr__
	sys.stdout = sys.__stdout__


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
