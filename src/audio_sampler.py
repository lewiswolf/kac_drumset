'''
This file defines an arbitrary audio sampler to be used as part of this project. This class
is intended to act as a template for any type of audio sampler, as the methods and properties
defined below, which are required for use with this system, will be inherited from this
parent class. This design also ensures that this project's dataset constructor (essentially
dataset.py) can be completely abstracted, and detached from its input.
'''

# core
from abc import ABC, abstractmethod
import math
from typing import Any, Union

# dependencies
import numpy as np				# maths
import numpy.typing as npt		# typing for numpy
import soundfile as sf			# audio read & write

# src
from settings import settings


class AudioSampler(ABC):
	'''
	Template parent class for an audio sample. The intended use when
	deployed:
		sampler = AudioSampler()
		for i in range(n):
			sampler.updateParameters()
			sampler.generateWaveform()
			x = sampler.waveform
			y = sampler.getLabels()
			sampler.export()
	'''

	# public
	duration: float = settings['data_length']				# duration of the audio file (seconds)
	sr: int = settings['sample_rate']						# sample rate (hz)
	length: int = math.ceil(duration * sr)					# length of the audio file (samples)
	waveform: npt.NDArray[np.float64] = np.zeros(length)	# the audio sample itself

	# private
	__sample_count: int = 0									# internal counter to track the amount of samples generated

	def __init__(self, **kwargs: Any) -> None:
		pass

	def export(self, absolutePath: str) -> None:
		'''
		Write the generated waveform to a file.
		'''
		sf.write(absolutePath, self.waveform, self.sr)

	def __countSamplesGenerated__(self) -> None:
		'''
		This method is used to update the sample counter. Its intended use is to be included
		in updateProperties, to keep the sampler agnostic of the loop that it functions within.
		'''
		self.__sample_count += 1

	@abstractmethod
	def generateWaveform(self) -> None:
		'''
		This method should be used to set self.waveform.
		'''
		pass

	@abstractmethod
	def getLabels(self) -> list[Union[float, int]]:
		'''
		This method should return the y labels for the generated audio.
		'''
		pass

	@abstractmethod
	def updateProperties(self) -> None:
		'''
		This method should be used to update the properties of the sampler
		when inside a generator loop.
		'''
		pass
