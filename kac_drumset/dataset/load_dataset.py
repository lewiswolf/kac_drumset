'''
This file contains the loadDataset method.
'''

# core
import json
import math
import os

# dependencies
from tqdm import tqdm			# CLI progress bar
import torch					# pytorch

# src
from .dataset import TorchDataset
from .input_representation import InputRepresentation
from .utils import tqdm_settings, listToTensor
from ..utils import printEmojis

__all__ = [
	'loadDataset',
]


def loadDataset(dataset_dir: str) -> TorchDataset:
	'''
	loadDataset imports a kac_drumset dataset from the directory specified by the absolute path dataset_dir.
	'''

	with open(os.path.normpath(f'{dataset_dir}/metadata.json')) as file:
		# import metadata
		file.readlines(1)
		dataset_size = int(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:-1])
		representation_settings = json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:-1])
		sampler = json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:-1])
		sampler_settings = json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:-1])
		file.readlines(1)
		# backwards compatibility
		# delete during a major upgrade
		if type(sampler) == str:
			sampler = {
				'name': sampler,
				'version': '1.1.0',
			}
		# create dataset
		dataset = TorchDataset(
			dataset_dir=dataset_dir,
			dataset_size=dataset_size,
			representation_settings=representation_settings,
			sampler=sampler,
			sampler_settings=sampler_settings,
			x_size=InputRepresentation.transformShape(
				math.ceil(sampler_settings['duration'] * sampler_settings['sample_rate']),
				representation_settings,
			),
		)
		# import loop
		printEmojis('Importing dataset... 📚')
		with tqdm(total=dataset_size, bar_format=tqdm_settings['bar_format'], unit=tqdm_settings['unit']) as bar:
			for i in range(dataset_size):
				# import relevant information
				file.readlines(1)
				x = torch.as_tensor(json.loads(
					file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:-1],
				))
				y = listToTensor(json.loads(
					file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:],
				))
				file.readlines(1)
				# append input features to dataset
				dataset.__setitem__(i, x, y)
				bar.update(1)
		file.close()
	return dataset
