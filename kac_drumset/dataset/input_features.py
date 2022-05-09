'''
This file is used to transform arrays of raw audio into various input types ready for training a neural network. This
file also includes other helper methods relating to the size of the input data and normalising audio.
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
	'InputFeatures',
	'SpectrogramSettings',
]

# necessary to enforce dtype throughout the project, see todo.md ->
# 'Internal types for nested lists, numpy arrays and pytorch tensors'
torch.set_default_dtype(torch.float64)


class SpectrogramSettings(TypedDict, total=False):
	'''
	These settings deal strictly with the input representations of the data. For FFT, this is calculated using the
	provided n_bins for the number of frequency bins, window_length and hop_length. The mel representation uses the same
	settings as the FFT, with the addition of n_mels, the number of mel frequency bins. And lastly, constant q transform
	makes use of only the number of bins, as well as the hop_length (there is no overlapping window functionality
	currently supported). n_bins is used to indicate the maximum amount of bins to be used, whereas in reality this is
	altered to ensure that there are an even amount of bins per octave.
	'''

	f_min: float											# minimum frequency of transform, cqt only (hz)
	hop_length: int											# hop length in samples
	n_bins: int												# number of frequency bins for the spectral density function
	n_mels: int												# number of mel frequency bins (used when INPUT_FEATURES == 'mel')
	window_length: int										# window length in samples


class InputFeatures():
	'''
	This class is used to convert a raw waveform into a user defined input representation, which includes the fourier
	transform, mel spectrogram and constant q transform. The intended use of this class when deployed:
		IF = InputFeatures()
		X = np.zeros((n,) + IF.transformShape(len(waveform))))
		for i in range(n):
			X[i] = IF.transform(waveform)
	'''

	bins_per_octave: int
	f_min: float
	feature_type: Literal['end2end', 'fft', 'mel']
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
		feature_type: Literal['end2end', 'fft', 'mel'],
		sr: int,
		normalise_input: bool = False,
		spectrogram_settings: SpectrogramSettings = {},
	) -> None:
		'''
		This method sets up the class ready to produce input features. Based on the type of output/spectrogram, the internal
		settings are also configured.
		'''

		# initialise default settings
		default_settings: SpectrogramSettings = {
			'f_min': 22.05,
			'n_bins': 512,
			'n_mels': 128,
			'window_length': 512,
			'hop_length': 256,
		}
		# initialise user defined variables
		default_settings.update(spectrogram_settings)
		spectrogram_settings = default_settings
		self.feature_type = feature_type
		self.normalise_input = normalise_input
		self.sr = sr

		# configure end2end
		if self.feature_type == 'end2end':
			self.transform = self.__end2end__

		# configure fft
		if self.feature_type == 'fft':
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
		if self.feature_type == 'mel':
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
