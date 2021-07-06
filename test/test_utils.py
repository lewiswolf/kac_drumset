# core
import cProfile
import math
import pstats
from typing import Callable, Literal, Union

# dependencies
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import soundfile as sf


class testTone():
	'''
	This class produces an arbitrary test tone, such as a sine wave or a sawtooth wave.
	params:
		hz 		- frequency of the sinewave
		length 	- duration of the sine wave in seconds
		sr 		- sample rate in hz
		type 	- type of waveform. currently supported = [sine, sawtooth]
	'''

	def __init__(self, hz: float, length: float, sr: int, waveform: Literal['saw', 'sin', 'sqr', 'tri'] = 'sin') -> None:
		self.hz = hz								# frequency
		self.sr = sr								# sample rate
		self.length = math.ceil(self.sr * length)	# duration of sine wave in samples
		self.type = waveform						# type of waveform
		self.wave = self.__generateWav(waveform)	# wave array

	def __generateWav(self, waveform: Literal['saw', 'sin', 'sqr', 'tri']) -> npt.NDArray[np.float64]:
		'''
		Render a specified waveform to a numpy array.
		'''
		if waveform == 'saw':
			return -2 / math.pi * np.arctan(1 / np.tan(math.pi * self.hz * (np.arange(self.length) / self.sr)))
		if waveform == 'sin':
			return np.sin(2 * math.pi * self.hz * (np.arange(self.length) / self.sr))
		if waveform == 'sqr':
			sin = np.sin(2 * math.pi * self.hz * (np.arange(self.length) / self.sr))
			return sin / np.abs(sin) 
		if waveform == 'tri':
			return 2 / math.pi * np.arcsin(np.sin(2 * math.pi * self.hz * (np.arange(self.length) / self.sr)))

	def exportWav(self, filepath: str) -> None:
		'''
		Write waveform to file.
		'''
		sf.write(filepath, self.wave, self.sr)


def plotSpectrogram(
	spectrogram: npt.NDArray[np.float64],
	sr: Union[int, None] = None,
	scale: float = 1.0,
	window_length: Union[int, None] = None,
	hop_length: Union[int, None] = None,
) -> None:
	'''
	Plots a spectrogram, using the settings of spectrogram to infer the axes.
	params:
		sr	 			- sample rate, in hz
		scale			- linear/logarithmic control for the y axis (default: 1.0, linear)
		window_length	- window length used as part of the spectral density function, in samples
		hop_length		- hop length used as part of the spectral density function, in samples
	'''

	fig, ax = plt.subplots(1, figsize=(12, 6), dpi=100)
	plt.imshow(spectrogram, aspect='auto', cmap='YlGn', origin='lower')
	cbar = plt.colorbar(location="bottom")
	cbar.ax.set_xlabel('Power')

	if window_length and sr:
		# default torchaudio value
		if not hop_length:
			hop_length = window_length // 2

		# map x axis from frames to seconds
		plt.xticks(
			np.linspace(0, spectrogram.shape[1], num=11),
			np.round(np.linspace(
				0,
				(hop_length / window_length) * window_length * spectrogram.shape[1] / sr,
				num=11,
			), decimals=2),
		)
		ax.set(xlabel='Time (Seconds)')
	else:
		ax.set(xlabel='Frames')

	if sr:
		# map y axis from bins to frequency spectrum
		plt.yticks(
			np.linspace(0, spectrogram.shape[0], num=11),
			np.round(np.power(np.linspace(0, 1, num=11), scale) * sr / 2).astype('int64'),
		)
		ax.set(ylabel='Frequency (Hz)')
	else:
		ax.set(ylabel='Frequency Bins')

	plt.show()


def plotWaveform(waveform: npt.NDArray[np.float64], sr: int) -> None:
	'''
	Plots a waveform using matplotlib. Designed to handle mono and multichannel inputs of
	shape [C, S] or [S], where C is the number of channels, and S is the number of samples.
	'''

	if waveform.ndim == 1:
		# render a mono waveform
		fig, ax = plt.subplots(1, figsize=(10, 1.75), dpi=100)
		ax.plot(np.linspace(0, len(waveform) / sr, num=len(waveform)), waveform, color='black')
		ax.set(xlabel='Time (Seconds)', ylabel='Amplitude')
	elif waveform.ndim == 2:
		# render a multi channel waveform
		fig, ax = plt.subplots(waveform.shape[0], 1, figsize=(10, waveform.shape[0] * 1.75), dpi=100, squeeze=False)
		time = np.linspace(0, len(waveform[0]) / sr, num=len(waveform[0]))
		for i, channel in enumerate(waveform):
			ax[i][0].plot(time, channel, color='black')
			ax[i][0].set(xlabel='Time (Seconds)', ylabel='Amplitude')
	else:
		raise ValueError('Incorrect size of input array; only [N * M] & [M] supported.')

	plt.tight_layout()
	plt.show()


def withProfiler(func: Callable, n: int) -> None:
	'''
	Calls the input function using cProfile to generate a performance report in the console.
	Prints the n most costly functions.
	'''

	with cProfile.Profile() as pr:
		func()
	stats = pstats.Stats(pr)
	stats.sort_stats(pstats.SortKey.TIME)
	stats.print_stats(n)
