# dependencies
import torch				# pytorch

# src
from .audio_sampler import SamplerSettings
from .input_representation import RepresentationSettings

__all__ = [
	'TorchDataset',
]


class TorchDataset(torch.utils.data.Dataset):
	'''
	Pytorch wrapper for the generated/loaded dataset. Formats the dataset's input data
	into a tensor self.X, and the labels, if present, into a tensor self.Y.
	'''

	X: torch.Tensor
	Y: torch.Tensor
	dataset_dir: str
	dataset_size: int
	representation_settings: RepresentationSettings
	sampler: str
	sampler_settings: SamplerSettings

	def __init__(
		self,
		x_size: tuple[int, ...],
		dataset_size: int,
		sampler: str,
		representation_settings: RepresentationSettings,
		sampler_settings: SamplerSettings,
	) -> None:
		'''
		Initialise dataset.
		'''
		self.X = torch.zeros((dataset_size,) + x_size)
		self.sampler = sampler
		self.representation_settings = representation_settings
		self.sampler_settings = sampler_settings

	def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
		'''
		Return the data and its labels, if they exist.
		'''
		return self.X[i], self.Y[i]

	def __len__(self) -> int:
		'''
		Return the dataset size.
		'''
		return self.X.shape[0]

	def __setitem__(self, i: int, x: torch.Tensor, y: torch.Tensor) -> None:
		'''
		Set self.X and self.Y at a specific index. If self.Y doesn't already exist,
		it is initialised here.
		'''
		# if self.Y doesn't yet exist, and a label is given, create the shape for Y
		if not hasattr(self, 'Y'):
			self.Y = torch.zeros((self.__len__(), ) + y.shape)
		# add data samples to self
		self.X[i] = x
		self.Y[i] = y
