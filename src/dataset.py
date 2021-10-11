'''
Methods for constructing and loading a pytorch dataset. The constructor function defined
below creates a dataset of audio files which are stored on disk, and configures the
metadata for the dataset. When loading a dataset, its internal settings are checked to
ensure that the project settings and the dataset settings align. If this is not the case,
the dataset is either ammended or, if necessary, regenerated entirely.
'''

# core
import json
import os
import sys
from typing import Literal, Type, TypedDict, Union

# dependencies
import click					# CLI arguments
import soundfile as sf			# audio read & write
import torch					# pytorch
from tqdm import tqdm			# CLI progress bar

# src
from audio_sampler import AudioSampler
from input_features import InputFeatures
from settings import settings, SpectroSettings


tqdm_settings = {
	'bar_format': '{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}, Elapsed: {elapsed}, ETA: {remaining}, {rate_fmt}  ',
	'unit': ' data samples',
	'total': settings['dataset_size'],
}

# necessary to enforce dtype throughout the project, see todo.md ->
# 'Internal types for nested lists, numpy arrays and pytroch tensors'
torch.set_default_dtype(torch.float64)


class SampleMetadata(TypedDict, total=False):
	'''
	Metadata format for each audio sample. Each audio sample consists of a wav file stored
	on disk alongside its respective input data (x) and labels (y). The implementation of
	this class assumes that the labels (y) are not always present.
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
	dataset_size: int		 								# how many data samples are there in the dataset?
	data_length: float										# length of each sample in the dataset (seconds)
	sample_rate: int										# audio sample rate (hz)
	input_features: Literal['end2end', 'fft', 'mel', 'cqt']	# how is the data represented when it is fed to the network?
	normalise_input: bool									# should each sample in the dataset be normalised before training?
	spectro_settings: SpectroSettings						# spectrogram settings
	data: list[SampleMetadata]								# the dataset itself


class TorchDataset(torch.utils.data.Dataset):
	'''
	Pytorch wrapper for the generated/loaded dataset. Formats the dataset's input data
	into a tensor self.X, and the labels, if present, into a tensor self.Y.
	'''

	X: torch.Tensor		# data
	Y: torch.Tensor		# labels

	def __init__(self, dataset_size: int, x_size: tuple[int, ...]) -> None:
		'''
		Initialise self.X.
		'''
		self.X = torch.zeros((dataset_size,) + x_size)

	def __getitem__(self, i: int) -> tuple[torch.Tensor, Union[torch.Tensor, None]]:
		'''
		Return the data and its labels, if they exist.
		'''
		if hasattr(self, 'Y'):
			return self.X[i], self.Y[i]
		else:
			return self.X[i], None

	def __len__(self) -> int:
		'''
		Return the dataset size.
		'''
		return self.X.shape[0]

	def setitem(self, i: int, x: torch.Tensor, y: torch.Tensor) -> None:
		'''
		Set self.X and self.Y at a specific index. If self.Y doesn't already exist,
		it is initialised here.
		'''
		# if self.Y doesn't yet exist, and a label is given, create the shape for Y
		if not hasattr(self, 'Y') and y.tolist():
			self.Y = torch.zeros((self.X.shape[0], ) + y.shape)
		# add data samples to self
		self.X[i] = x
		if hasattr(self, 'Y'):
			self.Y[i] = y


def parseMetadataToString() -> str:
	'''
	Parse the project metadata, as defined in DatasetMetadata, to a raw JSON string with
	line breaks.
	'''

	str = r'{'
	str += f'{os.linesep}'

	# This is a mypy compliant implementation of iterating over typeddict keys. A simpler
	# yet error inducing alternative would be to use:
	# for key in DatasetMetadata.__dict__['__annotations__'].keys():
	# 	print(settings[key])
	d_keys = DatasetMetadata.__dict__['__annotations__'].keys()
	for key, value in settings.items():
		if key in d_keys:
			str += rf'"{key}": {json.dumps(value)},{os.linesep}'

	str += rf'"data": [{os.linesep}'
	return str


def parseDataSampleToString(s: SampleMetadata, finalLine: bool) -> str:
	'''
	Parse a datasample, as defined by SampleMetadata, to a raw JSON string with line breaks.
	This function is designed to be implemented within a for loop.
	'''

	str = r'{'
	str += fr'{os.linesep}"filepath": "{s["filepath"]}",'
	str += fr'{os.linesep}"x": {s["x"]}'
	if 'y' in s and s['y']:
		str += fr',{os.linesep}"y": {s["y"]}'
	str += f'{os.linesep}'
	str += r'}]}' if finalLine else r'},' + f'{os.linesep}'
	return str


def generateDataset(
	DataSampler: Type[AudioSampler],
	dataset_size: int = settings['dataset_size'],
	dataset_dir: str = 'data',
) -> TorchDataset:
	'''
	Generates a dataset of audio samples. The generated dataset, including the individual
	.wav files and the metadata.json, are saved in the directory specified by dataset_dir,
	which is a path relative to the current working directory.
	'''

	cwd = os.getcwd()
	IF = InputFeatures()
	sampler = DataSampler()
	dataset = TorchDataset(dataset_size, IF.transformShape(sampler.length))

	# clear dataset folder
	for file in os.listdir(f'{cwd}/{dataset_dir}'):
		if file != '.gitignore':
			os.remove(f'{cwd}/{dataset_dir}/{file}')

	print(f'Generating dataset... {"🎯" if sys.platform in ["linux", "darwin"] else ""}')
	with open(f'{cwd}/{dataset_dir}/metadata.json', 'at') as new_file:
		new_file.write(parseMetadataToString())
		with tqdm(**tqdm_settings) as pbar:
			for i in range(dataset_size):
				# prepare sample
				sampler.generateWaveform()
				x = IF.transform(sampler.waveform)
				y = sampler.getLabels()
				# append input features to dataset
				dataset.setitem(i, x, torch.as_tensor(y))
				# bounce the raw audio
				relative_path = f'{dataset_dir}/sample_{i:05d}.wav'
				sampler.export(f'{cwd}/{relative_path}')
				# export metadata
				new_file.write(parseDataSampleToString({
					'filepath': relative_path,
					'x': x.tolist(),
					'y': y,
				}, i == dataset_size - 1))
				pbar.update(1)
		new_file.close()
	return dataset


# def loadDataset(DataSample: Type[AudioSample]) -> TorchDataset:
# 	'''
# 	Attempts to load a dataset if one has already been generated. Verifies the metadata
# 	of the loaded dataset, and ammends the dataset if necessary.
# 	'''

# 	cwd = os.getcwd()

# 	class DatasetIncompatible(Exception):
# 		pass

# 	try:
# 		with open(f'{cwd}/data/metadata.json') as file:
# 			# skip the initial '{'
# 			file.readlines(1)

# 			# confirm project settings match
# 			inputIncompatible = False
# 			for key in DatasetMetadata.__dict__['__annotations__'].keys():
# 				if key == 'data':
# 					file.readlines(1)
# 					break
# 				value = json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:-1])
# 				# TO FIX: see todo.md -> 'Extendable way to loop over TypedDict keys'
# 				if key == 'DATASET_SIZE' and settings[key] > value:
# 					raise DatasetIncompatible
# 				elif (key == 'SAMPLE_RATE' or key == 'DATA_LENGTH') and settings[key] != value:
# 					raise DatasetIncompatible
# 				elif settings[key] != value:
# 					inputIncompatible = True
# 			print("Preprocessing dataset... 📚")
# 			dataset = TorchDataset()

# 			# construct dataset from json
# 			if not inputIncompatible:
# 				with tqdm(**tqdm_settings) as pbar:
# 					for i in range(settings['DATASET_SIZE']):
# 						# import relevant information
# 						file.readlines(2)
# 						x = torch.as_tensor(
# 							json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:-1]),
# 						)
# 						y = torch.as_tensor(
# 							json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:]),
# 						)
# 						file.readlines(1)
# 						# on the initial run, infer the size of dataset.Y
# 						if i == 0:
# 							dataset.Y = torch.zeros((settings['DATASET_SIZE'],) + tuple(y.shape))
# 						# append input features to dataset
# 						dataset.setitem(i, x, y)
# 						pbar.update(1)
# 				file.close()

# 			# regenerate inputs features and metadata.json
# 			else:
# 				with open(f'{cwd}/data/metadata_temp.json', 'at') as newFile:
# 					newFile.write(parseMetadataToString())
# 					with tqdm(**tqdm_settings) as pbar:
# 						for i in range(settings['DATASET_SIZE']):
# 							# import relevant information
# 							file.readlines(1)
# 							relativePath = json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:-1])
# 							file.readlines(1)
# 							x = inputFeatures(sf.read(f'{cwd}/{relativePath}')[0])
# 							y = torch.as_tensor(
# 								json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:]),
# 							)
# 							file.readlines(1)
# 							# on the initial run, infer the size of dataset.Y
# 							if i == 0:
# 								dataset.Y = torch.zeros((settings['DATASET_SIZE'],) + tuple(y.shape))
# 							# append input features to dataset
# 							dataset.setitem(i, x, y)
# 							# export metadata
# 							newFile.write(parseDataSampleToString(relativePath, x, y, i == settings['DATASET_SIZE'] - 1))
# 							pbar.update(1)
# 					file.close()
# 					os.remove(f'{cwd}/data/metadata.json')
# 					os.rename(f'{cwd}/data/metadata_temp.json', f'{cwd}/data/metadata.json')
# 					newFile.close()
# 		return dataset

# 	# handle exceptions
# 	except Exception as e:
# 		if type(e).__name__ == 'DatasetIncompatible':
# 			print("Imported dataset is incompatible with the current project settings. 🤷")
# 			print("Check data/metadata.json to see your previous project setting.")
# 		else:
# 			print("Could not load a dataset. 🤷")
# 		if not click.confirm('Do you want to generate a new dataset?', default=None, prompt_suffix=': '):
# 			sys.exit()
# 		return generateDataset(DataSample)
