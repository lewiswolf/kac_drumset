# core
import json
import os
import sys
from typing import TypedDict

# dependencies
import click								# CLI arguments
import torch								# pytorch
from tqdm import tqdm						# CLI progress bar

# src
from settings import settings				# creates a project settings object
from input_features import inputFeatures	# methods for converting .wav files into a range of input features

# tests
sys.path.insert(1, os.path.join(os.getcwd(), 'test'))
import random
from test_utils import testTone


class DataSample(TypedDict):
	'''
	Metadata format for each data sample. Each data sample consist of a wav file stored
	on disk alongside its respective labels.
	'''
	filepath: str				# location of .wav file, relative to project directory
	labels: list[float]			# labels for each sample


class DatasetMetadata(TypedDict):
	'''
	This class is used when exporting and importing the metadata for the dataset. This
	object and the settings object are compared to ensure a loaded dataset matches the
	project settings.
	'''
	DATASET_SIZE: int		 	# how many data samples are there in the dataset?
	DATA_LENGTH: float			# length of each sample in the dataset (seconds)
	SAMPLE_RATE: int			# audio sample rate (hz)
	data: list[DataSample]		# the dataset itself


class TorchDataset(torch.utils.data.Dataset):
	'''
	Pytorch wrapper for the generated/loaded dataset.
	'''
	# TO ADD: better type hinting for pytorch, such that the internal datatype (float64,
	# float32, etc.) is specified. Alternative is to use a global `torch.set_default_dtype(d)`.

	def __init__(self, data: list[DataSample]) -> None:
		X, Y = [], []
		for sample in data:
			X.append(sample['filepath'])
			Y.append(sample['labels'])
		self.X = inputFeatures(X)
		self.Y = torch.tensor(Y)

	def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
		return self.X[index], self.Y[index]

	def __len__(self) -> int:
		return settings['DATASET_SIZE']


def generateDataset() -> TorchDataset:
	'''
	Generates a dataset of sounds. The generated dataset, including the individual .wav
	files and the metadata.json, is saved in ../data.
	'''

	metadata: DatasetMetadata = {
		'DATASET_SIZE': settings['DATASET_SIZE'],
		'DATA_LENGTH': settings['DATA_LENGTH'],
		'SAMPLE_RATE': settings['SAMPLE_RATE'],
		'data': [],
	}

	# clear dataset folder
	datasetFolder = os.path.join(os.getcwd(), 'data')
	for file in os.listdir(datasetFolder):
		if file != '.gitignore':
			os.remove(os.path.join(datasetFolder, file))

	# generate datatset
	print('Generating dataset... 🎯')
	with tqdm(
		bar_format='{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}, Elapsed: {elapsed}, ETA: {remaining}, {rate_fmt}  ',
		unit=' data samples',
		total=settings['DATASET_SIZE'],
	) as pbar:
		for i in range(settings['DATASET_SIZE']):
			# create a random test tone, export it as a wav, and append the metadata to the output
			filepath = f'data/sample_{i:05d}.wav'
			sin = testTone((random.random() * 770) + 110, settings['DATA_LENGTH'], settings['SAMPLE_RATE'])
			sin.exportWav(os.path.join(os.getcwd(), filepath))
			metadata['data'].append({
				"filepath": filepath,
				"labels": [sin.hz],
			})

			# update progress bar
			pbar.update(1)

	# export metadata json
	with open(os.path.join(os.getcwd(), 'data/metadata.json'), 'w') as json_file:
		json.dump(metadata, json_file, skipkeys=True, indent="\t")

	return TorchDataset(metadata['data'])


def loadDataset() -> TorchDataset:
	'''
	Attempts to load a dataset if one has already been generated. Verifies the metadata
	of the loaded dataset, and ammends the dataset if necessary.
	'''

	# custom exception if dataset generation settings do not match the project setttings
	class DatasetIncompatible(Exception):
		pass

	try:
		# load a dataset if it exists
		# TO ADD: it would be good to type check this at runtime, to check for errors such as KeyError
		# https://stackoverflow.com/questions/66665336 a non-hacky version of this?
		metadata: DatasetMetadata = json.load(open(os.path.join(os.getcwd(), 'data/metadata.json'), 'r'))

		# if the project settings and data settings do not align, throw error
		# TO ADD: make this loop over the metadata keys to make any future object expansions simpler.
		# https://github.com/python/mypy/issues/6262 `for key in dict.keys()` produces a type error.
		if (metadata['DATASET_SIZE'] < settings['DATASET_SIZE']
					or metadata['SAMPLE_RATE'] != settings['SAMPLE_RATE']
					or metadata['DATA_LENGTH'] != settings['DATA_LENGTH']):
			raise DatasetIncompatible

		# if the dataset is bigger than the project settings, trim its size, or simply return the datatset
		if metadata['DATASET_SIZE'] > settings['DATASET_SIZE']:
			return TorchDataset(metadata['data'][: settings['DATASET_SIZE']])
		else:
			return TorchDataset(metadata['data'])

	except (FileNotFoundError, KeyError):
		# generate new dataset if no dataset exists
		print('Could not load a dataset. 🤷')
		return generateDataset()
	except DatasetIncompatible:
		# if the dataset is incompatible with the current project settings, ask to regenerate
		print('Imported dataset is incompatible with the current project settings. 🤷')
		if not click.confirm('Do you want to generate a new dataset?', default=None, prompt_suffix=': '):
			print('Check data/metadata.json to see your previous project setting.')
			sys.exit()
		return generateDataset()
