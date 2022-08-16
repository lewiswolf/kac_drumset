'''
This file contains the generateDataset method.
'''

# core
import json
import os

# dependencies
from tqdm import tqdm			# CLI progress bar

# src
from .audio_sampler import AudioSampler, SamplerSettings
from .input_representation import InputRepresentation, RepresentationSettings
from .dataset import TorchDataset
from .utils import tqdm_settings, listToTensor
from ..utils import clearDirectory, printEmojis

__all__ = [
	'generateDataset',
]


def generateDataset(
	Sampler: type[AudioSampler],
	sampler_settings: SamplerSettings,
	dataset_dir: str = os.path.normpath(f'{os.path.dirname(__file__)}/../../data'),
	dataset_size: int = 10,
	representation_settings: RepresentationSettings = {},
) -> TorchDataset:
	'''
	Generates a dataset of audio samples. The generated dataset, including the individual .wav files and the metadata.json,
	are saved in the directory specified by the absolute filepath dataset_dir.
	'''

	# initialise classes
	IR = InputRepresentation(
		sampler_settings['sample_rate'],
		representation_settings,
	)
	sampler = Sampler(**sampler_settings)
	dataset = TorchDataset(
		dataset_dir=dataset_dir,
		dataset_size=dataset_size,
		representation_settings=IR.settings,
		sampler=Sampler.__name__,
		sampler_settings=sampler_settings,
		x_size=IR.transformShape(sampler.length, IR.settings),
	)
	# housekeeping
	clearDirectory(dataset_dir)
	printEmojis('Generating dataset... 🎯')
	# generation loop
	with open(
		os.path.normpath(f'{dataset_dir}/metadata.json'),
		'at',
	) as new_file:
		# add metadata
		new_file.write(r'{' + '\n')
		new_file.write(rf'"dataset_size": {dataset_size},' + '\n')
		new_file.write(rf'"representation_settings": {json.dumps(IR.settings)},' + '\n')
		new_file.write(rf'"sampler": "{Sampler.__name__}",' + '\n')
		new_file.write(rf'"sampler_settings": {json.dumps(sampler_settings)},' + '\n')
		# add data
		new_file.write(r'"data": [' + '\n')
		with tqdm(total=dataset_size, **tqdm_settings) as bar:
			for i in range(dataset_size):
				# prepare sample
				sampler.updateProperties(i)
				sampler.generateWaveform()
				x = IR.transform(sampler.waveform)
				y = sampler.getLabels()
				# append input features to dataset
				dataset.__setitem__(i, x, listToTensor(y))
				# bounce the raw audio
				sampler.export(f'{dataset_dir}/sample_{i:05d}.wav')
				# export metadata
				new_file.write(r'{' + '\n')
				new_file.write(rf'"x": {x.tolist()},' + '\n')
				new_file.write(rf'"y": {json.dumps(y)}' + '\n')
				new_file.write(r'}]}' if i == dataset_size - 1 else r'},' + '\n')
				bar.update(1)
		new_file.close()
	return dataset
