'''
This file is used to define and configure the project settings.
The configurable object is at the bottom of this file, whilst
the type declarations should be used as a guideline to ensure
that the settings object works as intended.
'''

# core
import sys
from typing import cast, Literal, TypedDict, Union

# dependencies
import pydantic 	# runtime type-checking


class PhysicalModelSettings(TypedDict):
	'''
	These settings deal strictly with physical model, a numerical method for
	generating the sounds of arbitrarily shaped drums.
	'''

	path_2_cuda: Union[str, None]							# absolute filepath to Nvidia's CUDA SDK
	allow_concave: bool										# are the drums allowed to be concave? or only convex?
	max_vertices: int										# maximum amount of vertices for a given drum


class SpectroSettings(TypedDict):
	'''
	These settings deal strictly with the input representations of the data.
	For FFT, this is calculated using the provided n_bins for the number of
	frequency bins, window_length and hop_length. The mel representation uses
	the same settings as the FFT, with the addition of n_mels, the number of
	mel frequency bins. And lastly, constant q transform makes use of only the
	number of bins, as well as the hop_length (there is no overlapping window
	functionality currently supported). n_bins is used to indicate the maximum
	amount of bins to be used, whereas in reality this is altered to ensure
	that there are an even amount of bins per octave.
	'''

	n_bins: int												# number of frequency bins for the spectral density function
	n_mels: int												# number of mel frequency bins (used when INPUT_FEATURES == 'mel')
	window_length: int										# window length in samples
	hop_length: int											# hop length in samples


class Settings(TypedDict):
	BATCH_SIZE: int											# bacth size used when training the network
	DATA_LENGTH: float										# length of each sample in the dataset (seconds)
	DATASET_SIZE: int										# how many data samples are there in the dataset?
	DATASET_SPLIT: tuple[float, float, float]				# size of the training, test and validation sets (must sum to 1.0)
	INPUT_FEATURES: Literal['end2end', 'fft', 'mel', 'cqt']	# how is the data represented when it is fed to the network?
	NORMALISE_INPUT: bool									# should each sample in the dataset be normalised before training?
	PM_SETTINGS: PhysicalModelSettings
	SAMPLE_RATE: int										# audio sample rate (hz)
	SPECTRO_SETTINGS: SpectroSettings


# the configurable object
settings: Settings = {
	'BATCH_SIZE': 4,
	'DATA_LENGTH': 5.0,
	'DATASET_SIZE': 10,
	'DATASET_SPLIT': (0.7, 0.15, 0.15),
	'INPUT_FEATURES': 'end2end',
	'NORMALISE_INPUT': False,
	'PM_SETTINGS': {
		'path_2_cuda': None,
		'allow_concave': True,
		'max_vertices': 10,
	},
	'SAMPLE_RATE': 44100,
	'SPECTRO_SETTINGS': {
		'n_bins': 512,
		'n_mels': 128,
		'window_length': 512,
		'hop_length': 256,
	},
}

# validate and enforce types at runtime
try:
	# TO FIX: see todo.md -> 'pydantic.create_model_from_typeddict has an incompatible type error'
	settings = cast(Settings, pydantic.create_model_from_typeddict(Settings)(**settings).dict())
except pydantic.ValidationError as e:
	print(f'ERROR: The project settings are not configured correctly. {e}')
	sys.exit()
