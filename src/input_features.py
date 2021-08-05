'''
This file is used to transform arrays of raw audio into various input types
ready for training a neural network. This file also includes other helper
methods relating to the size of the input data.
'''

# core
import math

# dependencies
import numpy as np 				# maths
import numpy.typing as npt		# typing for numpy
import librosa					# numpy audio manipulation
import torch					# pytorch
import torchaudio				# tensor audio manipulation

# src
from settings import SpectroSettings, settings
specSettings: SpectroSettings = settings['SPECTRO_SETTINGS']


# config settings specifically for use with a constant q transform
if settings['INPUT_FEATURES'] == 'cqt':
	f_min = 22.05
	max_octaves = math.log2((settings['SAMPLE_RATE'] * 0.5) / f_min)
	bins_per_octave = math.floor(specSettings['n_bins'] / max_octaves)
	n_bins = math.floor(bins_per_octave * max_octaves)


def inputSize() -> tuple[int, ...]:
	'''
	Helper method used for precomputing the size of an individual input feature.
	'''

	if settings['INPUT_FEATURES'] == 'end2end':
		return (math.ceil(settings['DATA_LENGTH'] * settings['SAMPLE_RATE']), )
	else:
		temporalWidth = math.ceil(
			(math.ceil(settings['DATA_LENGTH'] * settings['SAMPLE_RATE']) + 1) / specSettings['hop_length'],
		)
		if settings['INPUT_FEATURES'] == 'fft':
			return (specSettings['n_bins'] // 2 + 1, temporalWidth)
		if settings['INPUT_FEATURES'] == 'mel':
			return (specSettings['n_mels'], temporalWidth)
		if settings['INPUT_FEATURES'] == 'cqt':
			return (n_bins, temporalWidth)


def inputFeatures(waveform: npt.NDArray[np.float64]) -> torch.Tensor:
	'''
	This method produces the input features to pass to the neural network, by
	first importing a pre-generated .wav file, returning a tensor corresponding
	to a range of input features.
	'''

	# normalise the audio file
	if settings['NORMALISE_INPUT'] and np.max(waveform) != 1.0:
		waveform = waveform * (1.0 / np.max(waveform))

	# return the correct input representation as a tensor
	if settings['INPUT_FEATURES'] == 'end2end':
		return torch.as_tensor(waveform)

	if settings['INPUT_FEATURES'] == 'fft':
		return torchaudio.transforms.Spectrogram(
			n_fft=specSettings['n_bins'],
			win_length=specSettings['window_length'],
			hop_length=specSettings['hop_length'],
			power=2.0,
		)(torch.as_tensor(waveform))

	if settings['INPUT_FEATURES'] == 'mel':
		return torchaudio.transforms.MelSpectrogram(
			sample_rate=settings['SAMPLE_RATE'],
			n_mels=specSettings['n_mels'],
			n_fft=specSettings['n_bins'],
			win_length=specSettings['window_length'],
			hop_length=specSettings['hop_length'],
			power=2.0,
		)(torch.as_tensor(waveform))

	if settings['INPUT_FEATURES'] == 'cqt':
		# TO ADD: see todo.md -> 'Port librosa.vqt() to PyTorch'
		return torch.as_tensor(np.abs(librosa.cqt(
			waveform,
			sr=settings['SAMPLE_RATE'],
			n_bins=n_bins,
			bins_per_octave=bins_per_octave,
			fmin=f_min,
			hop_length=specSettings['hop_length'],
			dtype=np.float64,
		)))
