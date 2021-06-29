# core
import math

# dependencies
import numpy as np
import numpy.typing as npt
import soundfile as sf


class testTone():
	'''
	This class produces an arbitrary sine wave.
	params:
		hz - frequency of the sinewave
		length - duration of the sine wave in seconds
		sr - sample rate in hz
	'''

	def __init__(self, hz: float, length: float, sr: int) -> None:
		self.hz: float = hz											# frequency
		self.sr = sr												# sample rate
		self.length: int = math.ceil(self.sr * length)				# duration of sine wave in samples
		self.wave: npt.NDArray[np.float64] = self.__generateWav()	# wave array

	def __generateWav(self) -> npt.NDArray[np.float64]:
		'''
		Render a sine wave to a numpy array.
		'''

		two_pi: float = (2 * math.pi)
		sampleLength: float = 1 / self.sr
		phi: float = 0
		wave: npt.NDArray[np.float64] = np.zeros(self.length)

		for i in range(self.length):
			wave[i] = math.sin(phi)
			phi += two_pi * self.hz * sampleLength
			if phi > two_pi:
				phi -= math.floor(phi / two_pi) * two_pi

		return wave

	def exportWav(self, filepath: str) -> None:
		'''
		Write waveform to file.
		'''
		sf.write(filepath, self.wave, self.sr)
