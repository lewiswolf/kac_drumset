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
from .audio_sampler import AudioSampler, SamplerSettings
from .input_representation import InputRepresentation, RepresentationSettings
from .dataset import TorchDataset
from ..utils import clearDirectory, printEmojis

__all__ = [
	'generateDataset',
]


tqdm_settings = {
	'bar_format': '{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}, Elapsed: {elapsed}, ETA: {remaining}, {rate_fmt}  ',
	'unit': ' data samples',
}

# necessary to enforce dtype throughout the project, see todo.md ->
# 'Internal types for nested lists, numpy arrays and pytorch tensors'
torch.set_default_dtype(torch.float64)


def generateDataset(
	Sampler: Type[AudioSampler],
	sampler_settings: SamplerSettings,
	dataset_dir: str = os.path.normpath(f'{os.path.dirname(__file__)}/../../data'),
	dataset_size: int = 10,
	representation_settings: RepresentationSettings = {},
) -> TorchDataset:
	'''
	Generates a dataset of audio samples. The generated dataset, including the individual
	.wav files and the metadata.json, are saved in the directory specified by dataset_dir,
	which is a path relative to the current working directory.
	'''

	# initialise classes
	IR = InputRepresentation(
		sampler_settings['sample_rate'],
		representation_settings,
	)
	sampler = Sampler(**sampler_settings)
	dataset = TorchDataset(
		IR.transformShape(sampler.length),
		dataset_size,
		Sampler.__name__,
		representation_settings,
		sampler_settings,
	)

	# housekeeping
	clearDirectory(dataset_dir)
	printEmojis('Generating dataset... 🎯')

	# generation loop
	with open(
		os.path.normpath(f'{dataset_dir}/metadata.json'),
		'at',
	) as new_file:
		# initial line
		new_file.write(r'{' + f'{os.linesep}')
		# add metadata
		new_file.write(rf'"dataset_size": {dataset_size},{os.linesep}')
		new_file.write(rf'"sampler_settings": {json.dumps(sampler_settings)},{os.linesep}')
		new_file.write(rf'"representation_settings": {json.dumps(IR.settings)},{os.linesep}')
		# add data
		new_file.write(rf'"data": [{os.linesep}')

		with tqdm(total=dataset_size, **tqdm_settings) as bar:
			for i in range(dataset_size):
				# prepare sample
				# sampler.updateProperties()
				sampler.generateWaveform()
				x = IR.transform(sampler.waveform)
				y = sampler.getLabels()
				# append input features to dataset
				dataset.__setitem__(i, x, torch.as_tensor(y))
				# bounce the raw audio
				filepath = f'{dataset_dir}/sample_{i:05d}.wav'
				sampler.export(filepath)
				# export metadata
				new_file.write(r'{' + f'{os.linesep}')
				new_file.write(rf'"x": {x.tolist()},{os.linesep}')
				new_file.write(rf'"y": {y},{os.linesep}')
				new_file.write(r'}]}' if i == dataset_size - 1 else r'},' + f'{os.linesep}')
				bar.update(1)
		new_file.close()
	return dataset
