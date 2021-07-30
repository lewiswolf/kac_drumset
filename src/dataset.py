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
from typing import Literal, Type, TypedDict

# dependencies
import click					# CLI arguments
import soundfile as sf			# audio read & write
import torch					# pytorch
from tqdm import tqdm			# CLI progress bar

# src
from audio_sample import AudioSample
from input_features import inputFeatures, inputSize
from settings import settings, SpectroSettings


tqdmSettings = {
	'bar_format': '{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}, Elapsed: {elapsed}, ETA: {remaining}, {rate_fmt}  ',
	'unit': ' data samples',
	'total': settings['DATASET_SIZE'],
}


class SampleMetadata(TypedDict):
	'''
	Metadata format for each audio sample. Each audio sample consists of a wav file stored
	on disk alongside its respective input data (x) and labels (y).
	'''
	filepath: str											# location of .wav file, relative to project directory
	x: list													# input data for the network
	y: list													# labels for each sample


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
	a tensor self.Y, and the input into a tensor self.X.
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


def parseMetadataToString() -> str:
	'''
	Parse the project metadata, as defined in DatasetMetadata, to a raw JSON string with
	line breaks.
	'''

	str = r'{'
	str += f'{os.linesep}'
	for key in DatasetMetadata.__dict__['__annotations__'].keys():
		if key != 'data':
			# TO FIX: see todo.md -> 'Extendable way to loop over TypedDict keys'
			str += rf'"{key}": {json.dumps(settings[key])},{os.linesep}'
	str += rf'"data": [{os.linesep}'
	return str


def parseDataSampleToString(finalLine: bool, samplePath: str, x: torch.Tensor, y: torch.Tensor) -> str:
	'''
	Parse a datasample, as defined by SampleMetadata, to a raw JSON string with line breaks.
	This function is designed for use within a for loop.
	'''

	str = r'{'
	str += f'{os.linesep}'
	str += fr'"filepath": "{samplePath}",{os.linesep}'
	str += fr'"x": {x.tolist()},{os.linesep}'
	str += fr'"y": {y.tolist()}{os.linesep}'
	str += r'}]}' if finalLine else r'},' + f'{os.linesep}'
	return str


def generateDataset(DataSample: Type[AudioSample]) -> TorchDataset:
	'''
	Generates a dataset of audio samples. The generated dataset, including the individual
	.wav files and the metadata.json, are saved in ../data.
	'''

	cwd = os.getcwd()
	dataset = TorchDataset()

	# clear dataset folder
	for file in os.listdir(f'{cwd}/data'):
		if file != '.gitignore':
			os.remove(f'{cwd}/data/{file}')

	print('Generating dataset... 🎯')
	with open(f'{cwd}/data/metadata.json', 'at') as newFile:
		newFile.write(parseMetadataToString())
		with tqdm(**tqdmSettings) as pbar:
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
				newFile.write(parseDataSampleToString(i == settings['DATASET_SIZE'] - 1, relativePath, x, y))
				pbar.update(1)
		newFile.close()
	return dataset


def loadDataset(DataSample: Type[AudioSample]) -> TorchDataset:
	'''
	Attempts to load a dataset if one has already been generated. Verifies the metadata
	of the loaded dataset, and ammends the dataset if necessary.
	'''

	cwd = os.getcwd()

	class DatasetIncompatible(Exception):
		pass

	try:
		with open(f'{cwd}/data/metadata.json') as file:
			# skip the initial '{'
			file.readlines(1)

			# confirm project settings match
			inputIncompatible = False
			for key in DatasetMetadata.__dict__['__annotations__'].keys():
				if key == 'data':
					file.readlines(1)
					break
				value = json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:-1])
				# TO FIX: see todo.md -> 'Extendable way to loop over TypedDict keys'
				if key == 'DATASET_SIZE' and settings[key] > value:
					raise DatasetIncompatible
				elif (key == 'SAMPLE_RATE' or key == 'DATA_LENGTH') and settings[key] != value:
					raise DatasetIncompatible
				elif settings[key] != value:
					inputIncompatible = True

			print('Preprocessing dataset... 📚')
			dataset = TorchDataset()

			# construct dataset from json
			if not inputIncompatible:
				with tqdm(**tqdmSettings) as pbar:
					for i in range(settings['DATASET_SIZE']):
						# import relevant information
						file.readlines(2)
						x = torch.as_tensor(
							json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:-1]),
						)
						y = torch.as_tensor(
							json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:]),
						)
						file.readlines(1)
						# on the initial run, infer the size of dataset.Y
						if i == 0:
							dataset.Y = torch.zeros((settings['DATASET_SIZE'],) + tuple(y.shape))
						# append input features to dataset
						dataset.setitem(i, x, y)
						if i == settings['DATASET_SIZE'] - 1:
							file.close()
						pbar.update(1)

			# regenerate inputs features and metadata.json
			else:
				with open(f'{cwd}/data/metadata_temp.json', 'at') as newFile:
					newFile.write(parseMetadataToString())
					with tqdm(**tqdmSettings) as pbar:
						for i in range(settings['DATASET_SIZE']):
							# import relevant information
							file.readlines(1)
							relativePath = json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:-1])
							file.readlines(1)
							x = inputFeatures(sf.read(f'{cwd}/{relativePath}')[0])
							y = torch.as_tensor(
								json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:]),
							)
							file.readlines(1)
							# on the initial run, infer the size of dataset.Y
							if i == 0:
								dataset.Y = torch.zeros((settings['DATASET_SIZE'],) + tuple(y.shape))
							# append input features to dataset
							dataset.setitem(i, x, y)
							# export metadata
							newFile.write(parseDataSampleToString(i == settings['DATASET_SIZE'] - 1, relativePath, x, y))
							pbar.update(1)
					file.close()
					os.remove(f'{cwd}/data/metadata.json')
					os.rename(f'{cwd}/data/metadata_temp.json', f'{cwd}/data/metadata.json')
					newFile.close()
		return dataset

	# handle exceptions
	except Exception as e:
		if type(e).__name__ == 'DatasetIncompatible':
			print('Imported dataset is incompatible with the current project settings. 🤷')
			print('Check data/metadata.json to see your previous project setting.')
		else:
			print('Could not load a dataset. 🤷')
		if not click.confirm('Do you want to generate a new dataset?', default=None, prompt_suffix=': '):
			sys.exit()
		return generateDataset(DataSample)
