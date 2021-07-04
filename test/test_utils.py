# core
import math

# dependencies
import librosa.display
import matplotlib.pyplot as plt
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
		self.hz = hz								# frequency
		self.sr = sr								# sample rate
		self.length = math.ceil(self.sr * length)	# duration of sine wave in samples
		self.wave = self.__generateWav()			# wave array

	def __generateWav(self) -> npt.NDArray[np.float64]:
		'''
		Render a sine wave to a numpy array.
		'''
		return np.sin(2 * math.pi * self.hz * (1 / self.sr) * np.arange(self.length))

	def exportWav(self, filepath: str) -> None:
		'''
		Write waveform to file.
		'''
		sf.write(filepath, self.wave, self.sr)


def plotSpectrogram(waveform: npt.NDArray[np.float64]) -> None:
	pass


def plotWaveform(waveform: npt.NDArray[np.float64], sr: int) -> None:
	'''
	Plots a waveform using matplotlib. Designed to handle mono and multichannel inputs of
	shape [C, S] or [S], where C is the number of channels, and S is the number of samples.
	'''

	if waveform.ndim == 1:
		# render a mono waveform
		fig, ax = plt.subplots(1, figsize=(10, 1.75), dpi=100)
		time = np.linspace(0, len(waveform) / sr, num=len(waveform))
		ax.plot(time, waveform, color='black')
		ax.set(xlabel='Time (Seconds)', ylabel='Amplitude')
	else:
		# render a multi channel waveform
		fig, ax = plt.subplots(waveform.shape[0], 1, figsize=(10, waveform.shape[0] * 1.75), dpi=100, squeeze=False)
		time = np.linspace(0, len(waveform[0]) / sr, num=len(waveform[0]))
		for i, channel in enumerate(waveform):
			ax[i][0].plot(time, channel, color='black')
			ax[i][0].set(xlabel='Time (Seconds)', ylabel='Amplitude')

	plt.tight_layout()
	plt.show()
