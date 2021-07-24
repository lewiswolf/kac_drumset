# core
import cProfile
import math
import pstats
import random
from typing import Callable, Literal, Union

# dependencies
import matplotlib.pyplot as plt			# graphs
import numpy as np						# maths
import numpy.typing as npt				# typing for numpy

# src
from audio_sample import AudioSample


class TestTone(AudioSample):
	'''
	This class produces an arbitrary test tone, such as a sine wave.
	params:
		f0 		- Fundamental frequency in hz.
		type 	- Type of waveform. Currently supported = [sawtooth, sine, square, triangle].
	'''

	def init(self) -> None:
		self.f0: float = random.random() * 770 + 110
		self.type: Literal['saw', 'sin', 'sqr', 'tri'] = 'sin'

	def generateWaveform(self) -> npt.NDArray[np.float64]:
		''' Renders a specified waveform to a numpy array. '''
		if self.type == 'saw':
			return -2 / math.pi * np.arctan(1 / np.tan(math.pi * self.f0 * (np.arange(self.length) / self.sr)))
		if self.type == 'sin':
			return np.sin(2 * math.pi * self.f0 * (np.arange(self.length) / self.sr))
		if self.type == 'sqr':
			sin = np.sin(2 * math.pi * self.f0 * (np.arange(self.length) / self.sr))
			return sin / np.abs(sin)
		if self.type == 'tri':
			return 2 / math.pi * np.arcsin(np.sin(2 * math.pi * self.f0 * (np.arange(self.length) / self.sr)))


def plotSpectrogram(
	spectrogram: npt.NDArray[np.float64],
	input_type: Union[Literal['cqt', 'fft', 'mel'], None] = None,
	sr: Union[int, None] = None,
	hop_length: Union[int, None] = None,
	f_min: float = 20.0,
) -> None:
	'''
	Plots an arbitrary spectrogram, and formats the axes based on the type of spectrogram and
	the settings used for the spectral density function.

	params:
		input_type		- What type of spectrogram is it? Currently supported = [FFT, MelSpec, CQT]
		sr	 			- Audio sample rate in hz.
		hop_length		- Hop length used as part of the spectral density function, in samples.
		f_min			- Sets the minimum frequency of the y-axis in hz. This is necessary for an
						accurate representation of the cqt.
	'''

	fig, ax = plt.subplots(1, figsize=(12, 6), dpi=100)
	plt.imshow(spectrogram, aspect='auto', cmap='Greens', origin='lower')
	cbar = plt.colorbar(location="bottom")
	cbar.ax.set_xlabel('Power')

	# map x axis from frames to seconds
	if hop_length and sr:
		plt.xticks(
			np.linspace(0, spectrogram.shape[1], num=11),
			np.round(np.linspace(0, hop_length * spectrogram.shape[1] / sr, num=11), decimals=2),
		)
		ax.set(xlabel='Time (Seconds)')
	else:
		ax.set(xlabel='Frames')

	# map y axis from bins to frequency spectrum
	if input_type and sr:
		if input_type == 'cqt':
			y_ticks = np.linspace(0, spectrogram.shape[0], num=11)
			y_ticks = f_min * 2 ** (y_ticks / math.ceil(spectrogram.shape[0] / math.log2(sr * 0.5 / f_min)))
		if input_type == 'fft':
			y_ticks = np.linspace(0, sr * 0.5, num=11)
		if input_type == 'mel':
			y_ticks = np.linspace(0, math.log10(1 + sr * 0.5 / 700.0), num=11)
			y_ticks = 700.0 * (10.0 ** y_ticks - 1.0)

		plt.yticks(np.linspace(0, spectrogram.shape[0], num=11), np.round(y_ticks).astype('int64'))
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
		ax.plot(np.linspace(0, len(waveform) / sr, num=len(waveform)), waveform, color='green')
		ax.set(xlabel='Time (Seconds)', ylabel='Amplitude')
	elif waveform.ndim == 2:
		# render a multi channel waveform
		fig, ax = plt.subplots(waveform.shape[0], 1, figsize=(10, waveform.shape[0] * 1.75), dpi=100, squeeze=False)
		time = np.linspace(0, len(waveform[0]) / sr, num=len(waveform[0]))
		for i, channel in enumerate(waveform):
			ax[i][0].plot(time, channel, color='green')
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
