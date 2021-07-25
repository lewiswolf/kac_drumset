'''
This file defines an arbitrary audio sample to be used as part of this project. This class
is intended to act as a template for any type of audio sample, as the methods and properties
defined below, which are required for use with this system, will be inherited from this
parent class. This design also ensures that this project's dataset constructor (essentially
dataset.py) can be completely abstracted, and detached from its input.
'''

# core
import math

# dependencies
import numpy as np				# maths
import numpy.typing as npt		# typing for numpy
import soundfile as sf			# audio read & write

# src
from settings import settings


class AudioSample:
	''' Template parent class for an audio sample. '''
	def __init__(self):
		# default variables
		self.sr: int = settings['SAMPLE_RATE']
		self.length: int = math.ceil(settings['DATA_LENGTH'] * self.sr)
		self.y: list = []
		# call user defined init method
		self.init()
		# generate waveform and metadata
		self.wave: npt.NDArray[np.float64] = self.generateWaveform()

	def __export__(self, absolutePath: str) -> None:
		'''	Write the generated waveform to a file. '''
		sf.write(absolutePath, self.wave, self.sr)

	def init(self) -> None:
		''' template method '''
		pass

	def generateWaveform(self) -> npt.NDArray[np.float64]:
		''' template method '''
		return np.zeros(0)
