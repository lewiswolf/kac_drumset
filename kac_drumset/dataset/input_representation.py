'''
This file is used to transform arrays of raw audio into various input types ready for training a neural network. This
file also includes other helper methods such as calculating the size of the input data and normalising the input audio.
'''

# core
import math
from typing import Any, Literal, TypedDict

# dependencies
import numpy as np 				# maths
import numpy.typing as npt		# typing for numpy
import torch					# pytorch
import torchaudio				# tensor audio manipulation

__all__ = [
	'InputRepresentation',
	'SpectrogramSettings',
]

# necessary to enforce dtype throughout the project, see todo.md ->
# 'Internal types for nested lists, numpy arrays and pytorch tensors'
torch.set_default_dtype(torch.float64)


class SpectrogramSettings(TypedDict, total=False):
	'''
	These settings deal strictly with the input representations of the data. For FFT, this is calculated using the
	provided n_bins for the number of frequency bins, window_length and hop_length. The mel representation uses the same
	settings as the FFT, with the addition of n_mels, the number of mel frequency bins.
	'''

	f_min: float				# minimum frequency of the transform in hertz (mel only)
	hop_length: int				# hop length in samples
	n_bins: int					# number of frequency bins for the spectral density function
	n_mels: int					# number of mel frequency bins (mel only)
	window_length: int			# window length in samples


class InputRepresentation():
	'''
	This class is used to convert a raw waveform into a user defined input representation, which includes end2end, the
	fourier transform, and a mel spectrogram. The intended use of this class when deployed:
		IR = InputRepresentation()
		X = np.zeros((n,) + IR.transformShape(len(waveform))))
		for i in range(n):
			X[i] = IR.transform(waveform)
	'''

	# for an explanation of why this type is Any, see
	# todo.md => `Assigning class methods to class variables`.
	# __normalise__: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
	# transform: Callable[[Any, npt.NDArray[np.float64]], torch.Tensor]
	__normalise__: Any
	transform: Any

	bins_per_octave: int
	f_min: float
	hop_length: int
	n_bins: int
	n_mels: int
	representation_type: Literal['end2end', 'fft', 'mel']
	sr: int
	transformer: torch.nn.Module
	window_length: int

	def __init__(
		self,
		representation_type: Literal['end2end', 'fft', 'mel'],
		sr: int,
		normalise_input: bool = True,
		spectrogram_settings: SpectrogramSettings = {},
	) -> None:
		'''
		InputRepresentation works by creating a variably defined method self.transform. This method uses the input settings to
		generate the correct input representation of the data.
		'''

		# initialise default settings
		default_settings: SpectrogramSettings = {
			'f_min': 22.05,
			'hop_length': 256,
			'n_bins': 512,
			'n_mels': 128,
			'window_length': 512,
		}
		default_settings.update(spectrogram_settings)
		spectrogram_settings = default_settings
		# initialise user defined variables
		self.representation_type = representation_type
		self.sr = sr
		self.__normalise__ = self.normalise if normalise_input else lambda x: x
		# configure end2end
		if self.representation_type == 'end2end':
			self.transform = self.__end2end__
		# configure fft
		if self.representation_type == 'fft':
			self.hop_length = spectrogram_settings['hop_length']
			self.n_bins = spectrogram_settings['n_bins']
			self.window_length = spectrogram_settings['window_length']
			self.transformer = torchaudio.transforms.Spectrogram(
				hop_length=self.hop_length,
				n_fft=self.n_bins,
				power=2.0,
				win_length=self.window_length,
			)
			self.transform = self.__withTransformer__
		# configure mel
		if self.representation_type == 'mel':
			self.f_min = spectrogram_settings['f_min']
			self.hop_length = spectrogram_settings['hop_length']
			self.n_bins = spectrogram_settings['n_bins']
			self.n_mels = spectrogram_settings['n_mels']
			self.window_length = spectrogram_settings['window_length']
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

	def __end2end__(self, waveform: npt.NDArray[np.float64]) -> torch.Tensor:
		'''
		Convert a numpy array into a PyTorch tensor.
		'''

		return torch.as_tensor(self.__normalise__(waveform))

	def __withTransformer__(self, waveform: npt.NDArray[np.float64]) -> torch.Tensor:
		'''
		Calculate a spectrogram output using a transformer (torch.nn.Module).
		'''

		return self.transformer(torch.as_tensor(self.__normalise__(waveform)))

	@staticmethod
	def normalise(waveform: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
		'''
		Normalise an audio waveform, such that x ∈ [-1.0, 1.0]
		'''

		x_min = np.min(waveform)
		return 2.0 * (waveform - x_min) / (np.max(waveform) - x_min) - 1.0

	def transformShape(self, data_length: int) -> tuple[int, ...]:
		'''
		Helper method used for precomputing the shape of an individual input feature.
		params:
			data_length		Length of the audio file (samples).
		'''

		if self.representation_type == 'end2end':
			return (data_length, )
		else:
			temporalWidth = math.ceil((data_length + 1) / self.hop_length)
			if self.representation_type == 'fft':
				return (self.n_bins // 2 + 1, temporalWidth)
			if self.representation_type == 'mel':
				return (self.n_mels, temporalWidth)

	def settings(self) -> SpectrogramSettings:
		return {
			'f_min': self.f_min if hasattr(self, 'f_min') else 22.05,
			'hop_length': self.hop_length if hasattr(self, 'hop_length') else 256,
			'n_bins': self.n_bins if hasattr(self, 'n_bins') else 512,
			'n_mels': self.n_mels if hasattr(self, 'n_mels') else 128,
			'window_length': self.window_length if hasattr(self, 'window_length') else 512,
		}
