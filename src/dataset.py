# core
import json
import os
import sys
from typing import cast, Type, TypedDict

# dependencies
import click					# CLI arguments
import pydantic					# runtime type-checking
import torch					# pytorch
from tqdm import tqdm			# CLI progress bar

# src
from audio_sample import AudioSample, SampleMetadata
from input_features import inputFeatures
from settings import settings


class DatasetMetadata(TypedDict):
	'''
	This class is used when exporting and importing the metadata for the dataset. This
	object and the settings object are compared to ensure a loaded dataset matches the
	project settings.
	'''
	DATASET_SIZE: int		 	# how many data samples are there in the dataset?
	DATA_LENGTH: float			# length of each sample in the dataset (seconds)
	SAMPLE_RATE: int			# audio sample rate (hz)
	data: list[SampleMetadata]	# the dataset itself


class TorchDataset(torch.utils.data.Dataset):
	'''
	Pytorch wrapper for the generated/loaded dataset. Formats the dataset's labels into
	a tensor self.Y, and sends the data itself to be preprocessed into input features.
	'''

	def __init__(self, data: list[SampleMetadata]) -> None:
		X, Y = [], []
		for sample in data:
			X.append(sample['x'])
			Y.append(sample['y'])
		self.X = torch.tensor(X)
		self.Y = torch.tensor(Y)

	def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
		return self.X[i], self.Y[i]

	def __len__(self) -> int:
		return settings['DATASET_SIZE']


def generateDataset(DataSample: Type[AudioSample]) -> TorchDataset:
	'''
	Generates a dataset of audio samples. The generated dataset, including the individual
	.wav files and the metadata.json, are saved in ../data.
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
			sample = DataSample()
			sample.exportWAV(f'data/sample_{i:05d}.wav')
			# sample.metadata['x'] = inputFeatures(sample.wave)
			metadata['data'].append(sample.metadata)
			pbar.update(1)

	# export metadata json
	with open(os.path.join(os.getcwd(), 'data/metadata.json'), 'w') as json_file:
		json.dump(metadata, json_file, skipkeys=True, indent="\t")

	return TorchDataset(metadata['data'])


def loadDataset(DataSample: Type[AudioSample]) -> TorchDataset:
	'''
	Attempts to load a dataset if one has already been generated. Verifies the metadata
	of the loaded dataset, and ammends the dataset if necessary.
	'''

	# custom exception if dataset generation settings do not match the project setttings
	class DatasetIncompatible(Exception):
		pass

	try:
		# load a dataset if it exists
		metadata: DatasetMetadata = json.load(open(os.path.join(os.getcwd(), 'data/metadata.json'), 'r'))
		# validate and enforce types at runtime
		# TO FIX: see todo.md -> 'pydantic.create_model_from_typeddict has an incompatible type error'
		metadata = cast(DatasetMetadata, pydantic.create_model_from_typeddict(DatasetMetadata)(**metadata).dict())

		# if the project settings and data settings do not align, throw error
		# TO ADD: see todo.md -> 'Extendable way to loop over TypedDict keys'
		if (metadata['DATASET_SIZE'] < settings['DATASET_SIZE']
					or metadata['SAMPLE_RATE'] != settings['SAMPLE_RATE']
					or metadata['DATA_LENGTH'] != settings['DATA_LENGTH']):
			raise DatasetIncompatible

		# if the dataset is bigger than the project settings, trim its size, or return the entire atatset
		return TorchDataset(metadata['data'][: settings['DATASET_SIZE']])

	except FileNotFoundError:
		# generate new dataset if no dataset exists
		print('Could not load a dataset. 🤷')
		return generateDataset(DataSample)
	except (pydantic.ValidationError, DatasetIncompatible):
		# if the dataset is incompatible with the current project settings, ask to regenerate
		print('Imported dataset is incompatible with the current project settings. 🤷')
		if not click.confirm('Do you want to generate a new dataset?', default=None, prompt_suffix=': '):
			print('Check data/metadata.json to see your previous project setting.')
			sys.exit()
		return generateDataset(DataSample)
