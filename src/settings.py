'''
This file is used to define and configure the project settings.
'''

from typing import TypedDict


class Settings(TypedDict):
	NUM_OF_TARGETS: int		# How many data samples are there in the dataset?
	SAMPLE_RATE: int		# audio samplerate


# the configurable object
settings: Settings = {
	'NUM_OF_TARGETS': 10,
	'SAMPLE_RATE': 44100,
}
