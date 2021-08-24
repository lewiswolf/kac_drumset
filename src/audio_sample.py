'''
This file defines an arbitrary audio sample to be used as part of this project. This class
is intended to act as a template for any type of audio sample, as the methods and properties
defined below, which are required for use with this system, will be inherited from this
parent class. This design also ensures that this project's dataset constructor (essentially
dataset.py) can be completely abstracted, and detached from its input.
'''

# core
from abc import ABC, abstractmethod
import math
from typing import Union

# dependencies
import numpy as np				# maths
import numpy.typing as npt		# typing for numpy
import soundfile as sf			# audio read & write

# src
from settings import settings


class AudioSample(ABC):
	''' Template parent class for an audio sample. '''

	sr: int = settings['SAMPLE_RATE']						# sample rate
	duration: int = math.ceil(settings['DATA_LENGTH'] * sr)	# duration of the audio sample
	wave: npt.NDArray[np.float64]							# the audio sample itself
	y: list[Union[float, int]]								# metadata / labels

	def __init__(self, y: list[Union[float, int]] = []) -> None:
		self.y = y
		self.wave = self.generateWaveform()

	def __export__(self, absolutePath: str) -> None:
		'''	Write the generated waveform to a file. '''
		sf.write(absolutePath, self.wave, self.sr)

	@abstractmethod
	def generateWaveform(self) -> npt.NDArray[np.float64]:
		pass
