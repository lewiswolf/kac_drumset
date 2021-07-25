'''
Methods for constructing and loading a pytorch dataset. The constructor function
defined below creates a dataset of audio files which are stored on disk, and
configures the metadata for the dataset. When loading a dataset, its internal settings
are checked to ensure that the project settings and the dataset settings align. If this
is not the case, the dataset is either ammended or regenerated entirely.
'''

# core
import json
import os
import sys
from typing import cast, Literal, Type, TypedDict

# dependencies
import click					# CLI arguments
import pydantic					# runtime type-checking
import soundfile as sf			# audio read & write
import torch					# pytorch
from tqdm import tqdm			# CLI progress bar

# src
from audio_sample import AudioSample
from input_features import inputFeatures, inputSize
from settings import settings, SpectroSettings


class SampleMetadata(TypedDict):
	'''
	Metadata format for each audio sample. Each audio sample consists of a wav file stored
	on disk alongside its respective input data (x) and labels (y).
	'''
	filepath: str				# location of .wav file, relative to project directory
	x: list						# input data for the network
	y: list						# labels for each sample


class DatasetMetadata(TypedDict):
	'''
	This class is used when exporting and importing the metadata for the dataset. This
	object and the settings object are compared to ensure a loaded dataset matches the
	project settings.
	'''
	DATASET_SIZE: int		 								# how many data samples are there in the dataset?
	DATA_LENGTH: float										# length of each sample in the dataset (seconds)
	SAMPLE_RATE: int										# audio sample rate (hz)
	INPUT_FEATURES: Literal['end2end', 'fft', 'mel', 'cqt']	# how is the data represented when it is fed to the network?
	NORMALISE_INPUT: bool									# should each sample in the dataset be normalised before training?
	SPECTRO_SETTINGS: SpectroSettings						# spectrogram settings
	data: list[SampleMetadata]								# the dataset itself


class TorchDataset(torch.utils.data.Dataset):
	'''
	Pytorch wrapper for the generated/loaded dataset. Formats the dataset's labels into
	a tensor self.Y, and sends the data itself to be preprocessed into input features.
	'''

	def __init__(self) -> None:
		self.X: torch.Tensor = torch.zeros((settings['DATASET_SIZE'],) + inputSize())
		self.Y: torch.Tensor

	def setitem(self, i: int, x: torch.Tensor, y: torch.Tensor) -> None:
		self.X[i] = x
		self.Y[i] = y

	def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
		return self.X[i], self.Y[i]

	def __len__(self) -> int:
		return settings['DATASET_SIZE']


def generateDataset(DataSample: Type[AudioSample]) -> TorchDataset:
	'''
	Generates a dataset of audio samples. The generated dataset, including the individual
	.wav files and the metadata.json, are saved in ../data.
	'''

	cwd = os.getcwd()
	dataset = TorchDataset()
	metadata: DatasetMetadata = {
		'DATASET_SIZE': settings['DATASET_SIZE'],
		'DATA_LENGTH': settings['DATA_LENGTH'],
		'SAMPLE_RATE': settings['SAMPLE_RATE'],
		'INPUT_FEATURES': settings['INPUT_FEATURES'],
		'NORMALISE_INPUT': settings['NORMALISE_INPUT'],
		'SPECTRO_SETTINGS': settings['SPECTRO_SETTINGS'],
		'data': [],
	}

	# clear dataset folder
	for file in os.listdir(f'{cwd}/data'):
		if file != '.gitignore':
			os.remove(f'{cwd}/data/{file}')

	# generate datatset
	print('Generating dataset... 🎯')
	with open(f'{cwd}/data/metadata.json', 'a') as jsonFile:
		# add the initial section to metadata.json
		jsonFile.write(json.dumps(metadata)[: -2])

		with tqdm(
			bar_format='{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}, Elapsed: {elapsed}, ETA: {remaining}, {rate_fmt}  ',
			unit=' data samples',
			total=settings['DATASET_SIZE'],
		) as pbar:
			for i in range(settings['DATASET_SIZE']):
				# prepare sample
				sample = DataSample()
				x = inputFeatures(sample.wave)
				y = torch.tensor(sample.y)
				# on the initial run, infer the size of dataset.Y
				if i == 0:
					dataset.Y = torch.zeros((settings['DATASET_SIZE'],) + tuple(y.shape))
				# append input features to dataset
				dataset.setitem(i, x, y)
				# bounce the raw audio
				relativePath = f'data/sample_{i:05d}.wav'
				sample.__export__(f'{cwd}/{relativePath}')
				# export metadata
				jsonFile.write(r'{' + fr'"filepath": "{relativePath}", "x": {x.tolist()}, "y": {sample.y}')
				if i != settings['DATASET_SIZE'] - 1:
					jsonFile.write(r'}, ')
				else:
					jsonFile.write(r'}]}')
					jsonFile.close()
				pbar.update(1)

	return dataset


def loadDataset(DataSample: Type[AudioSample]) -> TorchDataset:
	'''
	Attempts to load a dataset if one has already been generated. Verifies the metadata
	of the loaded dataset, and ammends the dataset if necessary.
	'''

	print('Preprocessing dataset... 📚')
	cwd = os.getcwd()

	# custom exceptions used to handle changes made to the settings
	class DatasetIncompatible(Exception):
		pass

	class InputIncompatible(Exception):
		pass

	try:
		# load a dataset if it exists
		metadata: DatasetMetadata = json.load(open(f'{cwd}/data/metadata.json', 'r'))
		# validate and enforce types at runtime
		# TO FIX: see todo.md -> 'pydantic.create_model_from_typeddict has an incompatible type error'
		metadata = cast(DatasetMetadata, pydantic.create_model_from_typeddict(DatasetMetadata)(**metadata).dict())

		# confirm matching dataset settings
		# TO ADD: see todo.md -> 'Extendable way to loop over TypedDict keys'
		if (metadata['DATASET_SIZE'] < settings['DATASET_SIZE']
					or [metadata['SAMPLE_RATE'], metadata['DATA_LENGTH']]
					!= [settings['SAMPLE_RATE'], settings['DATA_LENGTH']]):
			raise DatasetIncompatible

		dataset = TorchDataset()
		dataset.Y = torch.zeros(
			(settings['DATASET_SIZE'],) + tuple(torch.as_tensor(metadata['data'][0]['y']).shape),
		)
		
		# confirm matching input settings
		# TO ADD: see todo.md -> 'Extendable way to loop over TypedDict keys'
		if ([metadata['INPUT_FEATURES'], metadata['NORMALISE_INPUT'], metadata['SPECTRO_SETTINGS']]
					!= [settings['INPUT_FEATURES'], settings['NORMALISE_INPUT'], settings['SPECTRO_SETTINGS']]):
			raise InputIncompatible

	# if no dataset exists
	except FileNotFoundError:
		print('Could not load a dataset. 🤷')
		if not click.confirm('Do you want to generate a new dataset?', default=None, prompt_suffix=': '):
			sys.exit()
		return generateDataset(DataSample)

	# if the dataset is incompatible with the current project settings
	except (pydantic.ValidationError, DatasetIncompatible):
		print('Imported dataset is incompatible with the current project settings. 🤷')
		if not click.confirm('Do you want to generate a new dataset?', default=None, prompt_suffix=': '):
			print('Check data/metadata.json to see your previous project setting.')
			sys.exit()
		return generateDataset(DataSample)

	# recompute input features if project settings don't align
	except InputIncompatible:
		# initialise new metadata.json
		os.remove(f'{cwd}/data/metadata.json')
		with open(f'{cwd}/data/metadata.json', 'a') as jsonFile:
			jsonFile.write(json.dumps({
				'DATASET_SIZE': settings['DATASET_SIZE'],
				'DATA_LENGTH': settings['DATA_LENGTH'],
				'SAMPLE_RATE': settings['SAMPLE_RATE'],
				'INPUT_FEATURES': settings['INPUT_FEATURES'],
				'NORMALISE_INPUT': settings['NORMALISE_INPUT'],
				'SPECTRO_SETTINGS': settings['SPECTRO_SETTINGS'],
				'data': [],
			})[: -2])

			with tqdm(
				bar_format='{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}, Elapsed: {elapsed}, ETA: {remaining}, {rate_fmt}  ',
				unit=' data samples',
				total=settings['DATASET_SIZE'],
			) as pbar:
				for i in range(settings['DATASET_SIZE']):
					# generate new input features
					relativePath = metadata['data'][i]['filepath']
					x = inputFeatures(sf.read(f'{cwd}/{relativePath}')[0])
					y = metadata['data'][i]['y']
					dataset.setitem(i, x, torch.as_tensor(y))

					# export metadata
					jsonFile.write(r'{' + fr'"filepath": "{relativePath}", "x": {x.tolist()}, "y": {y}')
					if i != settings['DATASET_SIZE'] - 1:
						jsonFile.write(r'}, ')
					else:
						jsonFile.write(r'}]}')
						jsonFile.close()
					pbar.update(1)

		return dataset

	# generate torch dataset if all checks are passed
	for i in range(settings['DATASET_SIZE']):
		dataset.setitem(
			i,
			torch.as_tensor(metadata['data'][i]['x']),
			torch.as_tensor(metadata['data'][i]['y']),
		)
	return dataset