'''
This file contains the regenerateDataPoints method.
'''

# core
import json
import os

# dependencies
from tqdm import tqdm			# CLI progress bar

# src
from .audio_sampler import AudioSampler
from .input_representation import InputRepresentation
from .dataset import TorchDataset
from .utils import tqdm_settings, listToTensor, tensorToList
from ..utils import printEmojis

__all__ = [
	'regenerateDataPoints',
]


def regenerateDataPoints(dataset: TorchDataset, Sampler: type[AudioSampler], entries: list[int]) -> TorchDataset:
	'''
	This method regenerates specific indices of a dataset.
	'''

	# handle errors
	if (not entries):
		return dataset
	if (Sampler.__name__ != dataset.sampler):
		raise TypeError(f'${Sampler.__name__} is not compatible with this dataset.')
	if (dataset.__len__() < max(entries) or min(entries) < 0):
		raise ValueError('Cannot replace all specified indices.')
	# sort entries, remove metadata, load generators
	entries.sort()
	os.remove(f'{dataset.dataset_dir}/metadata.json')
	IR = InputRepresentation(
		dataset.sampler_settings['sample_rate'],
		dataset.representation_settings,
	)
	sampler = Sampler(**dataset.sampler_settings)
	# generation loop
	printEmojis('Regenerating data points... 🛠')
	with open(
		os.path.normpath(f'{dataset.dataset_dir}/metadata.json'),
		'at',
	) as new_file:
		# add metadata
		new_file.write(r'{' + '\n')
		new_file.write(rf'"dataset_size": {dataset.__len__()},' + '\n')
		new_file.write(rf'"representation_settings": {json.dumps(dataset.representation_settings)},' + '\n')
		new_file.write(rf'"sampler": "{dataset.sampler}",' + '\n')
		new_file.write(rf'"sampler_settings": {json.dumps(dataset.sampler_settings)},' + '\n')
		# add data
		new_file.write(r'"data": [' + '\n')
		with tqdm(total=len(entries), **tqdm_settings) as bar:
			for i in range(dataset.__len__()):
				new_file.write(r'{' + '\n')
				if (entries != [] and entries[0] == i):
					# prepare new sample
					sampler.updateProperties(i)
					sampler.generateWaveform()
					x = IR.transform(sampler.waveform)
					y_list = sampler.getLabels()
					# append input features to dataset
					dataset.__setitem__(i, x, listToTensor(y_list))
					# delete old sample
					os.remove(f'{dataset.dataset_dir}/sample_{i:05d}.wav')
					sampler.export(f'{dataset.dataset_dir}/sample_{i:05d}.wav')
					# export metadata
					new_file.write(rf'"x": {x.tolist()},' + '\n')
					new_file.write(rf'"y": {json.dumps(y_list)}' + '\n')
					# count
					entries = entries[1:]
					bar.update(1)
				else:
					# export previously generated sample
					x, y_tensor = dataset.__getitem__(i)
					new_file.write(rf'"x": {x.tolist()},' + '\n')
					new_file.write(rf'"y": {json.dumps(tensorToList(y_tensor))}' + '\n')
				new_file.write(r'}]}' if i == dataset.__len__() - 1 else r'},' + '\n')
		new_file.close()
	return dataset
