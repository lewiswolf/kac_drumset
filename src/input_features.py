'''
This file is used to transform arrays of raw audio into various input types
ready for training a neural network. This file also includes other helper
methods relating to the size of the input data and normalising audio.
'''

# core
import math
from typing import Any, Literal

# dependencies
import numpy as np 				# maths
import numpy.typing as npt		# typing for numpy
import librosa					# numpy audio manipulation
import torch					# pytorch
import torchaudio				# tensor audio manipulation

# src
from settings import SpectroSettings, settings
spec_settings: SpectroSettings = settings['spectro_settings']


class InputFeatures():
	'''
	This class is used to convert a raw waveform into a user defined input
	representation, which includes the fourier transform, mel spectrogram
	and constant q transform. The intended use of this class when deployed:
		IF = InputFeatures()
		X = np.zeros((n,) + IF.transformShape(len(waveform))))
		for i in range(n):
			X[i] = IF.transform(waveform)
	'''

	bins_per_octave: int
	f_min: float
	feature_type: Literal['end2end', 'fft', 'mel', 'cqt']
	hop_length: int
	n_bins: int
	n_mels: int
	normalise_input: bool
	sr: int
	# for an explanation of why this type is Any, see
	# todo.md => `Assigning class methods to class variables`.
	# transform: Callable[[Any, npt.NDArray[np.float64]], torch.Tensor]
	transform: Any
	transformer: torch.nn.Module
	window_length: int

	def __init__(
		self,
		feature_type: Literal['end2end', 'fft', 'mel', 'cqt'] = settings['input_features'],
		normalise_input: bool = settings['normalise_input'],
		sr: int = settings['sample_rate'],
		f_min: float = spec_settings['f_min'],
		hop_length: int = spec_settings['hop_length'],
		n_bins: int = spec_settings['n_bins'],
		n_mels: int = spec_settings['n_mels'],
		window_length: int = spec_settings['window_length'],
	) -> None:
		'''
		This method sets up the class ready to produce input features. Based on the type
		of output/spectrogram, the internal settings are also configured.
		'''

		# initialise user defined variables
		self.feature_type = feature_type
		self.normalise_input = normalise_input
		self.sr = sr

		# configure end2end
		if self.feature_type == 'end2end':
			self.transform = self.__end2end__

		# configure fft
		if self.feature_type == 'fft':
			self.hop_length = hop_length
			self.n_bins = n_bins
			self.window_length = window_length
			self.transformer = torchaudio.transforms.Spectrogram(
				hop_length=self.hop_length,
				n_fft=self.n_bins,
				power=2.0,
				win_length=self.window_length,
			)
			self.transform = self.__withTransformer__

		# configure mel
		if self.feature_type == 'mel':
			# Necessary to fix a bug with mel spectrograms, see todo.md ->
			# 'Internal types for nested lists, numpy arrays and pytroch tensors'
			torch.set_default_dtype(torch.float64)
			self.f_min = f_min
			self.hop_length = hop_length
			self.n_bins = n_bins
			self.n_mels = n_mels
			self.window_length = window_length
			self.transformer = torchaudio.transforms.MelSpectrogram(
				f_min=self.f_min,
				hop_length=self.hop_length,
				n_fft=self.n_bins,
				n_mels=self.n_mels,
				power=2.0,
				sample_rate=self.sr,
				win_length=self.window_length,
			)
			self.transform = self.__withTransformer__

		# configure cqt
		if self.feature_type == 'cqt':
			self.f_min = f_min
			self.hop_length = hop_length
			# recalculate n_bins to form the closest match to the original kwarg
			max_octaves = math.log2((self.sr * 0.5) / self.f_min)
			self.bins_per_octave = math.floor(n_bins / max_octaves)
			self.n_bins = math.floor(self.bins_per_octave * max_octaves)
			self.transform = self.__cqt__

	def __cqt__(self, waveform: npt.NDArray[np.float64]) -> torch.Tensor:
		'''
		Calcualte the constant q transform using librosa.
		TO ADD: see todo.md -> 'Port librosa.vqt() to PyTorch'.
		'''
		waveform = self.__normalise__(waveform) if self.normalise_input else waveform
		return torch.as_tensor(np.abs(librosa.cqt(
			self.__normalise__(waveform),
			sr=self.sr,
			n_bins=self.n_bins,
			bins_per_octave=self.bins_per_octave,
			fmin=self.f_min,
			hop_length=self.hop_length,
			dtype=np.float64,
		)))

	def __end2end__(self, waveform: npt.NDArray[np.float64]) -> torch.Tensor:
		'''
		Convert a numpy array into a pytorch tensor.
		'''
		waveform = self.__normalise__(waveform) if self.normalise_input else waveform
		return torch.as_tensor(waveform)

	@staticmethod
	def __normalise__(waveform: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
		'''
		Normalise an audio waveform, such that x ∈ [-1.0, 1.0]
		'''
		x_min = np.min(waveform)
		return 2.0 * (waveform - x_min) / (np.max(waveform) - x_min) - 1.0

	def __withTransformer__(self, waveform: npt.NDArray[np.float64]) -> torch.Tensor:
		'''
		Calculate a spectrogram output using a transformer (torch.nn.Module).
		'''
		waveform = self.__normalise__(waveform) if self.normalise_input else waveform
		return self.transformer(torch.as_tensor(waveform))

	def transformShape(self, data_length: int) -> tuple[int, ...]:
		'''
		Helper method used for precomputing the shape of an individual input feature.
		params:
			data_length		Length of the audio file (samples).
		'''

		if self.feature_type == 'end2end':
			return (data_length, )
		else:
			temporalWidth = math.ceil((data_length + 1) / self.hop_length)
			if self.feature_type == 'fft':
				return (self.n_bins // 2 + 1, temporalWidth)
			if self.feature_type == 'mel':
				return (self.n_mels, temporalWidth)
			if self.feature_type == 'cqt':
				return (self.n_bins, temporalWidth)
