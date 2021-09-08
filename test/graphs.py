'''
This file contains functions used to produce multiple plots, either for testing or for export.
The styles for these graphs are defined in `./matplotlibrc`. Each function has the kwarg
export_path, which can be used to set the filepath and the image title for an exported image.
When export_path is falsey, the graph will simply be displayed on screen.
'''

# core
import math
import os
from typing import Literal, Union

# dependencies
import matplotlib.pyplot as plt			# graphs
import numpy as np						# maths
import numpy.typing as npt				# typing for numpy

# test
plt.style.use(f'{os.getcwd()}/test/matplotlibrc')


def plot1DMatrix(m: npt.NDArray[np.float64], export_path: str = '') -> None:
	'''
	A helper method for plotting a one dimensional matrix.
	'''

	# check size
	if m.ndim != 1:
		raise ValueError('Input matrix is not 1D')
	
	fig, ax = plt.subplots(1, figsize=(8, 1.75))
	ax.imshow(m.reshape((1, -1)))
	plt.xticks([])
	plt.yticks([])
	plt.savefig(export_path) if export_path else plt.show()


def plot2DMatrix(m: npt.NDArray, export_path: str = '') -> None:
	'''
	A helper method for plotting a two dimensional matrix, where
	(x0, y0) = M[N - 1, 0] for a matrix of size (N, M).
	'''

	# check size
	if m.ndim != 2:
		raise ValueError('Input matrix is not 2D')
	# plot matrix
	fig, ax = plt.subplots(1, figsize=(8, 8))
	ax.imshow(m)
	plt.xticks([])
	plt.yticks([])
	plt.savefig(export_path) if export_path else plt.show()


def plotPolygon(
	vertices: npt.NDArray[np.float64],
	centroid: Union[tuple[()], tuple[float, float]] = (),
	export_path: str = '',
) -> None:
	'''
	A helper method used to plot the vertices of a predefined polygon.
	params:
		vertices	A 2d numpy array of coordinates ([[x_1, y_1], [x_2, y_2], ...]), such that
					(x[i], y[i]) form a cartesian product. These coordinates should be ordered
					such that each pair (x[i], y[i]) is intended to be connected to both
					(x[i + 1], y[i + 1]) and (x[i - 1], y[i - 1]).
		centroid	A coordinate pair used to mark the geometric centroid of the shape.
	'''

	# plot polygon
	fig, ax = plt.subplots(1, figsize=(8, 8))
	plt.fill(vertices[:, 0], vertices[:, 1])
	if centroid:
		plt.scatter(*centroid, c=['#000000'], zorder=10)
	# set axes
	ax.set_xlim(np.min(vertices), np.max(vertices))
	ax.set_ylim(np.min(vertices), np.max(vertices))
	plt.axis('off')
	plt.xticks([])
	plt.yticks([])
	plt.savefig(export_path) if export_path else plt.show()


def plotSpectrogram(
	spectrogram: npt.NDArray[np.float64],
	input_type: Union[Literal['cqt', 'fft', 'mel'], None] = None,
	sr: Union[int, None] = None,
	hop_length: Union[int, None] = None,
	f_min: float = 20.0,
	export_path: str = '',
) -> None:
	'''
	Plots an arbitrary spectrogram, and formats the axes based on the type of spectrogram and the
	settings used for the spectral density function.

	params:
		input_type		What type of spectrogram is it? Currently supported = [FFT, MelSpec, CQT]
		sr	 			Audio sample rate in hz.
		hop_length		Hop length used as part of the spectral density function, in samples.
		f_min			Sets the minimum frequency of the y-axis in hz. This is necessary for an
						accurate representation of the cqt.
	'''

	# plot spectrogram
	fig, ax = plt.subplots(1, figsize=(12, 6))
	plt.imshow(spectrogram)
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

	plt.savefig(export_path) if export_path else plt.show()


def plotWaveform(waveform: npt.NDArray[np.float64], sr: int, export_path: str = '') -> None:
	'''
	Plots a waveform using matplotlib. Designed to handle mono and multichannel inputs of shape [C, S]
	or [S], where C is the number of channels, and S is the number of samples.
	'''

	if waveform.ndim == 1:
		# render a mono waveform
		fig, ax = plt.subplots(1, figsize=(10, 1.75))
		ax.plot(np.linspace(0, len(waveform) / sr, num=len(waveform)), waveform)
		# set axis labels
		ax.set(xlabel='Time (Seconds)', ylabel='Amplitude')
	elif waveform.ndim == 2:
		# render a multi channel waveform
		fig, ax = plt.subplots(waveform.shape[0], 1, figsize=(10, waveform.shape[0] * 1.75), squeeze=False)
		time = np.linspace(0, len(waveform[0]) / sr, num=len(waveform[0]))
		for i, channel in enumerate(waveform):
			ax[i][0].plot(time, channel)
		# set shared axis labels
		fig.add_subplot(111, frameon=False)
		plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
		plt.xlabel('Time (Seconds)')
		plt.ylabel('Amplitude')
	else:
		raise ValueError('Incorrect size of input array; only [N * M] & [M] supported.')

	plt.savefig(export_path) if export_path else plt.show()
