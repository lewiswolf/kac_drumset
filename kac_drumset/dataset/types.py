# core
from typing import Literal, TypedDict

# dependencies
import torch				# pytorch

__all__ = [
	'DatasetSettings',
	'SampleMetadata',
	'TorchDataset',
]


class DatasetSettings(TypedDict, total=False):
	'''
	'''

	dataset_dir: str
	dataset_size: int
	normalise_input: bool
	representation_type: Literal['end2end', 'fft', 'mel']


class SampleMetadata(TypedDict, total=False):
	'''
	Metadata format for each audio sample. Each audio sample consists of a wav file stored
	on disk alongside its respective input data (x) and labels (y). The implementation of
	this class assumes that the labels (y) are not always present.
	'''
	filepath: str											# location of .wav file, relative to project directory
	x: list													# input data for the network
	y: list													# labels for each sample


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
