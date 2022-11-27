'''
This file contains the transformDataset method.
'''

# core
import json
import math
import os

# dependencies
import soundfile as sf			# audio read & write
from tqdm import tqdm			# CLI progress bar
import torch					# pytorch

# src
from .dataset import TorchDataset
from .input_representation import InputRepresentation, RepresentationSettings
from .utils import tqdm_settings, tensorToList
from ..utils import printEmojis

__all__ = [
	'transformDataset',
]


def transformDataset(dataset: TorchDataset, representation_settings: RepresentationSettings) -> TorchDataset:
	'''
	transformDataset is used to transform the input representation of a loaded dataset. This method rewrites the
	metadata.json for the dataset, such that the dataset will be loaded with the new settings upon future use.
	'''

	# create new input representation
	IR = InputRepresentation(
		dataset.sampler_settings['sample_rate'],
		representation_settings,
	)
	# check that the dataset actually needs transforming
	if IR.settings == dataset.representation_settings:
		return dataset
	# remove metadata and dataset.X
	os.remove(f'{dataset.dataset_dir}/metadata.json')
	dataset.representation_settings = IR.settings
	dataset.X = torch.zeros((dataset.__len__(),) + IR.transformShape(
		math.ceil(dataset.sampler_settings['duration'] * dataset.sampler_settings['sample_rate']),
		IR.settings,
	))
	# generation loop
	printEmojis('Transforming dataset... 🛠')
	with open(
		os.path.normpath(f'{dataset.dataset_dir}/metadata.json'),
		'at',
	) as new_file:
		# add metadata
		new_file.write(r'{' + '\n')
		new_file.write(rf'"dataset_size": {dataset.__len__()},' + '\n')
		new_file.write(rf'"representation_settings": {json.dumps(IR.settings)},' + '\n')
		new_file.write(rf'"sampler": "{dataset.sampler}",' + '\n')
		new_file.write(rf'"sampler_settings": {json.dumps(dataset.sampler_settings)},' + '\n')
		# add data
		new_file.write(r'"data": [' + '\n')
		with tqdm(total=dataset.__len__(), bar_format=tqdm_settings['bar_format'], unit=tqdm_settings['unit']) as bar:
			for i in range(dataset.__len__()):
				# process data
				x = IR.transform(sf.read(f'{dataset.dataset_dir}/sample_{i:05d}.wav')[0])
				y = dataset.__getitem__(i)[1]
				dataset.__setitem__(i, x, y)
				# export metadata
				new_file.write(r'{' + '\n')
				new_file.write(rf'"x": {x.tolist()},' + '\n')
				new_file.write(rf'"y": {json.dumps(tensorToList(y))}' + '\n')
				new_file.write(r'}]}' if i == dataset.__len__() - 1 else r'},' + '\n')
				bar.update(1)
		new_file.close()
	return dataset
