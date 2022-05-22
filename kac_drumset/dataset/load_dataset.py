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
from .utils import tqdm_settings
from ..utils import printEmojis

__all__ = [
	'loadDataset',
]


class DatasetIncompatible(Exception):
	'''
	This exception is rasied when a loaded dataset cannot be used alongside
	the current project settings. The intended solution for this exception
	is to generate a completely new dataset.
	'''
	def __init__(self, e: Exception) -> None:
		printEmojis(f'🤷 Could not load a dataset due to a {type(e).__name__}.\n')


def loadDataset(dataset_dir: str = os.path.normpath(f'{os.path.dirname(__file__)}/../../data')) -> TorchDataset:

	with open(f'{dataset_dir}/metadata.json') as file:
		try:
			# import metadata
			file.readlines(1)
			dataset_size = int(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:-1])
			representation_settings = json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:-1])
			sampler = file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:-1]
			sampler_settings = json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:-1])
			file.readlines(1)

			dataset = TorchDataset(
				dataset_size=dataset_size,
				representation_settings=representation_settings,
				sampler=sampler,
				sampler_settings=sampler_settings,
				x_size=InputRepresentation.transformShape(
					math.ceil(sampler_settings['duration'] * sampler_settings['sample_rate']),
					representation_settings,
				),
			)

			printEmojis('Importing dataset... 📚')
			with tqdm(total=dataset_size, **tqdm_settings) as bar:
				for i in range(dataset_size):
					# import relevant information
					file.readlines(1)
					x = torch.as_tensor(json.loads(
						file.readlines(1)[0].replace(os.linesep, '').split(':', 1)[1][1:-1],
					))
					y = torch.as_tensor(json.loads(
						file.readlines(1)[0].replace(os.linesep, '').split(':', 1)[1][1:-1],
					))
					file.readlines(1)
					# append input features to dataset
					dataset.__setitem__(i, x, y)
					bar.update(1)
			file.close()

		except Exception as e:
			raise DatasetIncompatible(e)

	return dataset
