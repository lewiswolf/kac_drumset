'''
This file is used to define and configure the project settings.
The configurable object is at the bottom of this file, whilst
the type declarations should be used as a guideline to ensure
that the settings object works as intended.
'''

from typing import Literal, TypedDict


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
	DATASET_SIZE: int										# how many data samples are there in the dataset?
	DATA_LENGTH: float										# length of each sample in the dataset (seconds)
	SAMPLE_RATE: int										# audio sample rate (hz)
	INPUT_FEATURES: Literal['end2end', 'fft', 'mel', 'cqt']	# how is the data represented when it is fed to the network?
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
		'n_bins': 512,
		'n_mels': 128,
		'window_length': 512,
		'hop_length': 256,
	},
}
