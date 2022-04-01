# core
from abc import ABC, abstractmethod
import math
import struct
from typing import Union
import wave

# dependencies
import numpy as np				# maths
import numpy.typing as npt		# typing for numpy

__all__ = [
	'AudioSampler',
]


class AudioSampler(ABC):
	'''
	Template parent class for an audio sampler. The intended use when deployed:
		sampler = AudioSampler()
		for i in range(n):
			sampler.updateParameters()
			sampler.generateWaveform()
			x = sampler.waveform
			y = sampler.getLabels()
			sampler.export()
	'''

	duration: float						# duration of the audio file (seconds)
	sr: int								# sample rate (hz)
	length: int							# length of the audio file (samples)
	waveform: npt.NDArray[np.float64]	# the audio sample itself

	def __init__(self, duration: float, sr: int) -> None:
		'''
		Initialise sampler.
		'''
		self.duration = duration
		self.sr = sr
		self.length = math.ceil(duration * sr)
		self.waveform = np.zeros(self.length)

	def export(self, absolutePath: str) -> None:
		'''
		Write the generated waveform to a file.
		'''

		wav_format = (struct.pack('<i', int(s * (2 ** 23 - 1))) for s in self.waveform)
		with wave.open(absolutePath, 'w') as wav:
			wav.setnchannels(1)
			wav.setsampwidth(3)
			wav.setframerate(self.sr)
			for byte in wav_format:
				wav.writeframes(byte[0:3])
			wav.close()

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

	# @abstractmethod
	# def updateProperties(self) -> None:
	# 	'''
	# 	This method should be used to update the properties of the sampler
	# 	when inside a generator loop.
	# 	'''
	# 	pass

	# private
	# self._sample_count = 0
	# private
	# _sample_count: int					# internal counter to track the amount of samples generated
