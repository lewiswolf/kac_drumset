'''
'''

# core
import json
import os
from typing import Type

# dependencies
from tqdm import tqdm			# CLI progress bar
import torch					# pytorch

# src
from .input_representation import InputRepresentation, SpectrogramSettings
from .types import DatasetSettings, TorchDataset
from .utils import parseDataSampleToString
from ..sampler import AudioSampler, SamplerSettings
from ..utils import clearDirectory, printEmojis

__all__ = [
	'generateDataset',
]


tqdm_settings = {
	'bar_format': '{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}, Elapsed: {elapsed}, ETA: {remaining}, {rate_fmt}  ',
	'unit': ' data samples',
}

# necessary to enforce dtype throughout the project, see todo.md ->
# 'Internal types for nested lists, numpy arrays and pytroch tensors'
torch.set_default_dtype(torch.float64)


def generateDataset(
	Sampler: Type[AudioSampler],
	sampler_settings: SamplerSettings,
	dataset_settings: DatasetSettings = {},
	spectrogram_settings: SpectrogramSettings = {},
) -> TorchDataset:
	'''
	Generates a dataset of audio samples. The generated dataset, including the individual
	.wav files and the metadata.json, are saved in the directory specified by dataset_dir,
	which is a path relative to the current working directory.
	'''

	# initialise default settings
	default_settings: DatasetSettings = {
		'dataset_dir': os.path.normpath(f'{os.path.dirname(__file__)}/../../data'),
		'dataset_size': 10,
		'normalise_input': True,
		'representation_type': 'end2end',
	}
	default_settings.update(dataset_settings)
	dataset_settings = default_settings

	# initialise classes
	IR = InputRepresentation(
		normalise_input=dataset_settings['normalise_input'],
		representation_type=dataset_settings['representation_type'],
		spectrogram_settings=spectrogram_settings,
		sr=sampler_settings['sr'],
	)
	sampler = Sampler(**sampler_settings)
	dataset = TorchDataset(
		dataset_settings['dataset_size'],
		IR.transformShape(sampler.length),
	)

	# housekeeping
	clearDirectory(f'{dataset_settings["dataset_dir"]}')
	printEmojis('Generating dataset... 🎯')

	# generation loop
	with open(
		os.path.normpath(f'{dataset_settings["dataset_dir"]}/metadata.json'),
		'at',
	) as new_file:
		# initial line
		new_file.write(r'{' + f'{os.linesep}')
		# add metadata
		new_file.write(rf'"dataset_settings": {json.dumps(dataset_settings)},{os.linesep}')
		new_file.write(rf'"sampler_settings": {json.dumps(sampler_settings)},{os.linesep}')
		new_file.write(rf'"spectrogram_settings": {json.dumps(IR.settings())},{os.linesep}')
		# add data
		new_file.write(rf'"data": [{os.linesep}')

		with tqdm(total=dataset_settings['dataset_size'], **tqdm_settings) as bar:
			for i in range(dataset_settings['dataset_size']):
				# prepare sample
				# sampler.updateProperties()
				sampler.generateWaveform()
				x = IR.transform(sampler.waveform)
				y = sampler.getLabels()
				# append input features to dataset
				dataset.__setitem__(i, x, torch.as_tensor(y))
				# bounce the raw audio
				filepath = f'{dataset_settings["dataset_dir"]}/sample_{i:05d}.wav'
				sampler.export(filepath)
				# export metadata
				new_file.write(parseDataSampleToString({
					'filepath': filepath,
					'x': x.tolist(),
					'y': y,
				}, i == dataset_settings['dataset_size'] - 1))
				bar.update(1)
		new_file.close()
	return dataset
