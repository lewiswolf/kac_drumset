'''
This file is used to define and configure the project settings.
The configurable object is at the bottom of this file, whilst
the type declarations should be used as a guideline to ensure
that the settings object works as intended.
'''

from typing import Literal, TypedDict, Union


class SpectroSettings(TypedDict):
	n_fft: int												# number of fft bins
	n_mel: int												# number of mel frequency bins (used when INPUT_FEATURES == 'mel')
	window_length: Union[int, None]							# window length in samples
	hop_length: Union[int, None]							# hop length in samples


class Settings(TypedDict):
	DATASET_SIZE: int										# how many data samples are there in the dataset?
	DATA_LENGTH: float										# length of each sample in the dataset (seconds)
	SAMPLE_RATE: int										# audio sample rate (hz)
	INPUT_FEATURES: Literal['end2end', 'fft', 'mel', 'q']	# how is the data represented when it is fed to the network?
	NORMALISE_INPUT: bool									# should each sample in the dataset be normalised before training?
	SPECTRO_SETTINGS: SpectroSettings


# the configurable object
settings: Settings = {
	'DATASET_SIZE': 10,
	'DATA_LENGTH': 5.0,
	'SAMPLE_RATE': 44100,
	'INPUT_FEATURES': 'end2end',
	'NORMALISE_INPUT': False,
	'SPECTRO_SETTINGS': {
		'n_fft': 440,
		'n_mel': 128,
		'window_length': None,
		'hop_length': None,
	},
}
