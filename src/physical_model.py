# core
from typing import TypedDict


class DataSample(TypedDict):
	'''
	Metadata format for each data sample. Each data sample consist of a wav file stored
	on disk alongside its respective labels.
	'''
	filepath: str				# location of .wav file, relative to project directory
	labels: list[float]			# labels for each sample
