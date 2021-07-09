'''
This file is used to define and configure the project settings.
The configurable object is at the bottom of this file, whilst
the type declarations should be used as a guideline to ensure
that the settings object works as intended.
'''

from typing import Literal, TypedDict


class MelSettings(TypedDict):
	n_mels: int												# number of mel frequency bins (used when INPUT_FEATURES == 'mel')
	n_bins: int												# number of frequency bins for the STFT
	window_length: int										# window length in samples
	hop_length: int											# hop length in samples


class FFTSettings(TypedDict):
	n_bins: int												# number of frequency bins for the STFT
	window_length: int										# window length in samples
	hop_length: int											# hop length in samples


class Settings(TypedDict):
	DATASET_SIZE: int										# how many data samples are there in the dataset?
	DATA_LENGTH: float										# length of each sample in the dataset (seconds)
	SAMPLE_RATE: int										# audio sample rate (hz)
	NORMALISE_INPUT: bool									# should each sample in the dataset be normalised before training?
	INPUT_FEATURES: Literal['end2end', 'fft', 'mel', 'cqt']	# how is the data represented when it is fed to the network?
	FFT_SETTINGS: FFTSettings


# the configurable object
settings: Settings = {
	'DATASET_SIZE': 10,
	'DATA_LENGTH': 5.0,
	'SAMPLE_RATE': 44100,
	'NORMALISE_INPUT': False,
	'INPUT_FEATURES': 'cqt',
	'FFT_SETTINGS': {
		'n_bins': 800,
		'window_length': 400,
		'hop_length': 200,
	},
}
