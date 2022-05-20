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
	'RepresentationSettings',
]

# necessary to enforce dtype throughout the project, see todo.md ->
# 'Internal types for nested lists, numpy arrays and pytorch tensors'
torch.set_default_dtype(torch.float64)


class RepresentationSettings(TypedDict, total=False):
	'''
	These settings deal strictly with the input representations of the data. For FFT, this is calculated using the
	provided n_bins for the number of frequency bins, window_length and hop_length. The mel representation uses the same
	settings as the FFT, with the addition of n_mels, the number of mel frequency bins.
	'''

	f_min: float					# minimum frequency of the transform in hertz (mel only)
	hop_length: int					# hop length in samples
	n_bins: int						# number of frequency bins for the spectral density function
	n_mels: int						# number of mel frequency bins (mel only)
	normalise_input: bool			# should the input be normalised
	output_type: Literal[	# representation type
		'end2end',
		'fft',
		'mel',
	]
	window_length: int				# window length in samples


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
	settings: RepresentationSettings
	transformer: torch.nn.Module

	def __init__(
		self,
		sr: int,
		settings: RepresentationSettings = {},
	) -> None:
		'''
		InputRepresentation works by creating a variably defined method self.transform. This method uses the input settings to
		generate the correct input representation of the data.
		'''

		# initialise default settings
		default_settings: RepresentationSettings = {
			'f_min': 22.05,
			'hop_length': 256,
			'n_bins': 512,
			'n_mels': 32,
			'normalise_input': True,
			'output_type': 'end2end',
			'window_length': 512,
		}
		default_settings.update(settings)
		self.settings = default_settings
		self.sr = sr
		self.__normalise__ = self.normalise if self.settings['normalise_input'] else lambda x: x
		# configure end2end
		if self.settings['output_type'] == 'end2end':
			self.transform = self.__end2end__
		# configure fft
		if self.settings['output_type'] == 'fft':
			self.transformer = torchaudio.transforms.Spectrogram(
				hop_length=self.settings['hop_length'],
				n_fft=self.settings['n_bins'],
				power=2.0,
				win_length=self.settings['window_length'],
			)
			self.transform = self.__withTransformer__
		# configure mel
		if self.settings['output_type'] == 'mel':
			self.transformer = torchaudio.transforms.MelSpectrogram(
				f_min=self.settings['f_min'],
				hop_length=self.settings['hop_length'],
				n_fft=self.settings['n_bins'],
				n_mels=self.settings['n_mels'],
				power=2.0,
				sample_rate=self.sr,
				win_length=self.settings['window_length'],
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

		if self.settings['output_type'] == 'end2end':
			return (data_length, )
		else:
			temporalWidth = math.ceil((data_length + 1) / self.settings['hop_length'])
			if self.settings['output_type'] == 'fft':
				return (self.settings['n_bins'] // 2 + 1, temporalWidth)
			if self.settings['output_type'] == 'mel':
				return (self.settings['n_mels'], temporalWidth)
