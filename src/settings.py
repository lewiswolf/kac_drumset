'''
This file is used to define and configure the project settings.
The configurable object is at the bottom of this file, whilst
the type declarations should be used as a guideline to ensure
that the settings object works as intended.
'''

# core
import sys
from typing import cast, Literal, TypedDict

# dependencies
import pydantic 	# runtime type-checking


class PhysicalModelSettings(TypedDict):
	'''
	These settings deal strictly with physical model, a numerical method for
	generating the sounds of arbitrarily shaped drums.
	'''

	allow_concave: bool										# are the drums allowed to be concave? or only convex?
	decay_time: float										# how long will the simulation take to decay?
	drum_size: float										# size of the drum, spanning both the horizontal and vertical axes (m)
	material_density: float 								# material density of the simulated drum membrane (kg/m^2)
	max_vertices: int										# maximum amount of vertices for a given drum
	tension: float											# tension at rest (N/m)


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

	f_min: float											# minimum frequency of transform, cqt only (hz)
	n_bins: int												# number of frequency bins for the spectral density function
	n_mels: int												# number of mel frequency bins (used when INPUT_FEATURES == 'mel')
	window_length: int										# window length in samples
	hop_length: int											# hop length in samples


class Settings(TypedDict):
	batch_size: int											# bacth size used when training the network
	data_length: float										# length of each sample in the dataset (seconds)
	dataset_size: int										# how many data samples are there in the dataset?
	dataset_split: tuple[float, float, float]				# size of the training, test and validation sets (must sum to 1.0)
	input_features: Literal['end2end', 'fft', 'mel', 'cqt']	# how is the data represented when it is fed to the network?
	normalise_input: bool									# should each sample in the dataset be normalised before training?
	numba_path_2_cuda: str									# absolute filepath to Nvidia's CUDA SDK for use with numba
	pm_settings: PhysicalModelSettings
	sample_rate: int										# audio sample rate (hz)
	spectro_settings: SpectroSettings


# the configurable object
settings: Settings = {
	'batch_size': 4,
	'data_length': 1.5,
	'dataset_size': 10,
	'dataset_split': (0.7, 0.15, 0.15),
	'input_features': 'end2end',
	'normalise_input': False,
	'numba_path_2_cuda': '',
	'pm_settings': {
		'allow_concave': True,
		'decay_time': 1.5,
		'drum_size': 0.3,
		'material_density': 0.26,
		'max_vertices': 10,
		'tension': 2000.0,
	},
	'sample_rate': 44100,
	'spectro_settings': {
		'f_min': 22.05,
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
