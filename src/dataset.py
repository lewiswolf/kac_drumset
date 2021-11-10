'''
Methods for constructing and loading a pytorch dataset. The constructor function defined
below creates a dataset of audio files which are stored on disk, and configures the
metadata for the dataset. When loading a dataset, its internal settings are checked to
ensure that the project settings and the dataset settings align. If this is not the case,
the dataset is either ammended or, if necessary, regenerated entirely.
'''

# core
import json
# import math
import os
import shutil
import sys
# from typing import Any, Literal, TextIO, Type, TypedDict, Union
from typing import Any, Literal, Type, TypedDict, Union

# dependencies
# import click					# CLI arguments
# import soundfile as sf			# audio read & write
import torch					# pytorch
from tqdm import tqdm			# CLI progress bar

# src
from audio_sampler import AudioSampler
from input_features import InputFeatures
# from settings import settings, Settings, SpectroSettings
from settings import settings, SpectroSettings


tqdm_settings = {
	'bar_format': '{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}, Elapsed: {elapsed}, ETA: {remaining}, {rate_fmt}  ',
	'unit': ' data samples',
}

# necessary to enforce dtype throughout the project, see todo.md ->
# 'Internal types for nested lists, numpy arrays and pytroch tensors'
torch.set_default_dtype(torch.float64)


# class DatasetIncompatible(Exception):
# 	'''
# 	This exception is rasied when a loaded dataset cannot be used alongside
# 	the current project settings. The intended solution for this exception
# 	is to generate a completely new dataset.
# 	'''
# 	pass


# class InputIncompatible(Exception):
# 	'''
# 	'''
# 	pass


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
	project settings. The function confirmDatasetSettings relies on the order in which
	each property is defined below.
	'''
	data_length: float										# length of each sample in the dataset (seconds)
	dataset_size: int		 								# how many data samples are there in the dataset?
	input_features: Literal['end2end', 'fft', 'mel', 'cqt']	# how is the data represented when it is fed to the network?
	normalise_input: bool									# should each sample in the dataset be normalised before training?
	sample_rate: int										# audio sample rate (hz)
	spectro_settings: SpectroSettings						# spectrogram settings
	sampler_settings: dict[str, Any]						# keyword arguments used to set the audio sampler
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


# def confirmDatasetSettings(
# 	file: TextIO,
# 	sampler_settings: dict[str, Any],
# 	s: Settings = settings,
# ) -> None:
# 	'''
# 	This function was abstracted from loadDataset() specifically to be tested seperately.
# 	This function creates an instance of DatasetMetadata using the current project settings
# 	and checks whether the imported dataset matches these settings. Under certain conditions,
# 	the DatasetIncompatible error will we raised. This causes the loop to be exited, and the
# 	metadata.json file to be closed. In the event that the InputIncompatible error is raised,
# 	the imported dataset neds to amended before use with this project. To accomodate for this,
# 	the exception is only raised after reading through all of the metadata, to allow for
# 	seamless continuation when reading the metadata.json file.
# 	'''

# 	# to avoid typing errors whilst using a typed dict, we create an uncast instance of DatasetMetadata
# 	d_keys = DatasetMetadata.__dict__['__annotations__'].keys()
# 	current_settings = dict([(key, value) for key, value in s.items() if key in d_keys])
# 	current_settings['sampler_settings'] = sampler_settings
# 	# skip the initial '{'
# 	file.readlines(1)
# 	# confirmation loop
# 	input_incompatible = False
# 	for key in d_keys:
# 		if key == 'data':
# 			file.readlines(1)
# 			break
# 		value = json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:-1])

# 		# checks
# 		if key in ['dataset_size', 'data_length', 'sample_rate', 'sampler_settings']:
# 			if (key == 'dataset_size' and value > current_settings[key]) or value != current_settings[key]:
# 				file.close()
# 				raise DatasetIncompatible()
# 		else:
# 			if value != current_settings[key]:
# 				input_incompatible = True

# 	# break out from the main
# 	if input_incompatible:
# 		raise InputIncompatible()


def parseMetadataToString(dataset_size: int = settings['dataset_size'], sampler_settings: dict[str, Any] = {}) -> str:
	'''
	Parse the project metadata, as defined in DatasetMetadata, to a raw JSON string with
	line breaks.
	'''

	# initial line
	str = r'{' + f'{os.linesep}'

	# append information from the settings object
	# this is a mypy compliant implementation of iterating over typeddict keys. A simpler
	# yet error inducing alternative would be to use:
	# for key in DatasetMetadata.__dict__['__annotations__'].keys():
	# 	print(settings[key])
	d_keys = DatasetMetadata.__dict__['__annotations__'].keys()
	for key, value in settings.items():
		if key in d_keys:
			str += rf'"{key}": {json.dumps(value) if key != "dataset_size" else dataset_size},{os.linesep}'

	# add any sampler settings
	str += rf'"sampler_settings": {json.dumps(sampler_settings)},{os.linesep}'

	# add data
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
	sampler_settings: dict[str, Any] = {},
) -> TorchDataset:
	'''
	Generates a dataset of audio samples. The generated dataset, including the individual
	.wav files and the metadata.json, are saved in the directory specified by dataset_dir,
	which is a path relative to the current working directory.
	'''

	cwd = os.getcwd()
	IF = InputFeatures()
	sampler = DataSampler(**sampler_settings)
	dataset = TorchDataset(dataset_size, IF.transformShape(sampler.length))

	# clear dataset folder
	directory = f'{cwd}/{dataset_dir}'
	for file in os.listdir(directory):
		path = f'{directory}/{file}'
		if os.path.isdir(path):
			shutil.rmtree(path)
		elif file != '.gitignore':
			os.remove(path)

	print(f'Generating dataset... {"" if sys.platform not in ["linux", "darwin"] else "🎯"}')
	with open(
		os.path.normpath(f'{cwd}/{dataset_dir}/metadata.json'),
		'at',
	) as new_file:
		new_file.write(parseMetadataToString(dataset_size=dataset_size, sampler_settings=sampler_settings))
		with tqdm(total=dataset_size, **tqdm_settings) as pbar:
			for i in range(dataset_size):
				# prepare sample
				sampler.updateProperties()
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


# def loadDataset(
# 	DataSample: Type[AudioSampler],
# 	dataset_dir: str = 'data',
# 	sampler_settings: dict[str, Any] = {},
# ) -> TorchDataset:
# 	'''
# 	Attempts to load a dataset if one has already been generated. Verifies the metadata
# 	of the loaded dataset, and ammends the dataset if necessary.
# 	'''

# 	cwd = os.getcwd()
# 	IF = InputFeatures()
# 	dataset = TorchDataset(
# 		settings['dataset_size'],
# 		IF.transformShape(math.ceil(settings['data_length'] * settings['sample_rate'])),
# 	)

# 	try:
# 		with open(f'{cwd}/{dataset_dir}/metadata.json') as file:
# 			try:
# 				# raise appropriate exceptions if metadata doesn't match
# 				confirmDatasetSettings(file, sampler_settings)

# 				# create dataset entirely from metadata
# 				print(f'Importing dataset... {"" if sys.platform not in ["linux", "darwin"] else "📚"}')
# 				# with tqdm(**tqdm_settings) as pbar:
# 				# 	for i in range(settings['DATASET_SIZE']):
# 				# 		# import relevant information
# 				# 		file.readlines(2)
# 				# 		x = torch.as_tensor(
# 				# 			json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:-1]),
# 				# 		)
# 				# 		y = torch.as_tensor(
# 				# 			json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:]),
# 				# 		)
# 				# 		file.readlines(1)
# 				# 		# on the initial run, infer the size of dataset.Y
# 				# 		if i == 0:
# 				# 			dataset.Y = torch.zeros((settings['DATASET_SIZE'],) + tuple(y.shape))
# 				# 		# append input features to dataset
# 				# 		dataset.setitem(i, x, y)
# 				# 		pbar.update(1)
# 				# file.close()
# 				return dataset

# 			except Exception as e:
# 				if type(e).__name__ == 'InputIncompatible':
# 					print(f'Preprocessing dataset... {"" if sys.platform not in ["linux", "darwin"] else "📚"}')
# 					print(file.readlines(1))
# 				else:
# 					raise e

# 			# TO FIX: see todo.md -> 'Extendable way to loop over TypedDict keys'
# 			# if key == 'DATASET_SIZE' and settings[key] > value:
# 			# 	raise DatasetIncompatible
# 			# elif (key == 'SAMPLE_RATE' or key == 'DATA_LENGTH') and settings[key] != value:
# 			# 	raise DatasetIncompatible
# 			# elif settings[key] != value:
# 			# 	inputIncompatible = True

# # 			print(f'Preprocessing dataset... {"" if sys.platform not in ["linux", "darwin"] else "📚"}')
# # 			dataset = TorchDataset()

# # 			# construct dataset from json
# # 			if not inputIncompatible:
# # 				with tqdm(**tqdm_settings) as pbar:
# # 					for i in range(settings['DATASET_SIZE']):
# # 						# import relevant information
# # 						file.readlines(2)
# # 						x = torch.as_tensor(
# # 							json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:-1]),
# # 						)
# # 						y = torch.as_tensor(
# # 							json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:]),
# # 						)
# # 						file.readlines(1)
# # 						# on the initial run, infer the size of dataset.Y
# # 						if i == 0:
# # 							dataset.Y = torch.zeros((settings['DATASET_SIZE'],) + tuple(y.shape))
# # 						# append input features to dataset
# # 						dataset.setitem(i, x, y)
# # 						pbar.update(1)
# # 				file.close()

# # 			# regenerate inputs features and metadata.json
# # 			else:
# # 				with open(f'{cwd}/data/metadata_temp.json', 'at') as newFile:
# # 					newFile.write(parseMetadataToString())
# # 					with tqdm(**tqdm_settings) as pbar:
# # 						for i in range(settings['DATASET_SIZE']):
# # 							# import relevant information
# # 							file.readlines(1)
# # 							relativePath = json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:-1])
# # 							file.readlines(1)
# # 							x = inputFeatures(sf.read(f'{cwd}/{relativePath}')[0])
# # 							y = torch.as_tensor(
# # 								json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:]),
# # 							)
# # 							file.readlines(1)
# # 							# on the initial run, infer the size of dataset.Y
# # 							if i == 0:
# # 								dataset.Y = torch.zeros((settings['DATASET_SIZE'],) + tuple(y.shape))
# # 							# append input features to dataset
# # 							dataset.setitem(i, x, y)
# # 							# export metadata
# # 							newFile.write(parseDataSampleToString(relativePath, x, y, i == settings['DATASET_SIZE'] - 1))
# # 							pbar.update(1)
# # 					file.close()
# # 					os.remove(f'{cwd}/data/metadata.json')
# # 					os.rename(f'{cwd}/data/metadata_temp.json', f'{cwd}/data/metadata.json')
# # 					newFile.close()
# # 		return dataset

# # 	# handle exceptions
# 	except Exception as e:
# 		# print helpful info
# 		if type(e).__name__ == 'DatasetIncompatible':
# 			print(
# 				f'{"" if sys.platform not in ["linux", "darwin"] else "🤷 "}'
# 				'Imported dataset is incompatible with the current project settings.\n'
# 				'Check data/metadata.json to see your previous project setting.'
# 			)
# 		else:
# 			print(
# 				f'{"" if sys.platform not in ["linux", "darwin"] else "🤷 "}'
# 				f'Could not load a dataset due to a {type(e).__name__}.'
# 			)

# 		# regenerate dataset or exit
# 		if not click.confirm('Do you want to generate a new dataset?', default=None, prompt_suffix=': '):
# 			sys.exit()
# 		return generateDataset(DataSample, sampler_settings=sampler_settings)
