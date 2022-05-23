'''
This file is used to define a dataset class compatible with PyTorch.
'''

# dependencies
import torch				# pytorch

# src
from .audio_sampler import SamplerSettings
from .input_representation import RepresentationSettings

__all__ = [
	'TorchDataset',
]


class TorchDataset(torch.utils.data.Dataset):
	''' PyTorch wrapper for a dataset. '''

	dataset_dir: str									# dataset directory
	representation_settings: RepresentationSettings		# settings for InputRepresentation
	sampler: str										# the name of the sampler used to generate the dataset
	sampler_settings: SamplerSettings					# settings for the sampler
	X: torch.Tensor										# data
	Y: torch.Tensor										# labels

	def __init__(
		self,
		dataset_dir: str,
		dataset_size: int,
		representation_settings: RepresentationSettings,
		sampler: str,
		sampler_settings: SamplerSettings,
		x_size: tuple[int, ...],
	) -> None:
		''' Initialise dataset. '''
		self.dataset_dir = dataset_dir
		self.representation_settings = representation_settings
		self.sampler = sampler
		self.sampler_settings = sampler_settings
		self.X = torch.zeros((dataset_size,) + x_size)

	def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
		''' Return the data and its labels at index i. '''
		if not hasattr(self, 'Y'):
			raise ValueError('Dataset contains no data.')
		return self.X[i], self.Y[i]

	def __len__(self) -> int:
		''' Return the dataset size. '''
		return self.X.shape[0]

	def __setitem__(self, i: int, x: torch.Tensor, y: torch.Tensor) -> None:
		''' Set self.X and self.Y at a specific index. If self.Y does not already exist, it is initialised here. '''
		if not hasattr(self, 'Y'):
			self.Y = torch.zeros((self.__len__(), ) + y.shape)
		self.X[i] = x
		self.Y[i] = y
