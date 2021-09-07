'''
This file is used to prepare a dataset for multiple purposes. Given a threeway split,
this file will return three subsets prepared for either training, testing or evaluating
a neural network.
'''

# core
from itertools import accumulate

# dependencies
import torch		# pytorch

# src
from dataset import TorchDataset
from settings import settings


def inferSubdivisions(dataset_size: int, split: tuple[float, float, float]) -> list[int]:
	'''
	Calculates the integer splits for the training, testing and validation sets.
	'''

	subdivisions = [round(dataset_size * p) for p in split]
	# correct errors
	# this correction supposes that split[0] > split[1 or 2]
	subdivisions[0] += dataset_size - sum(subdivisions)
	# cumulative sums
	return list(accumulate(subdivisions))


def getTrainingDatasets(
	dataset: TorchDataset,
	batch_size: int = settings['batch_size'],
	split: tuple[float, float, float] = settings['dataset_split'],
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
	'''
	Extracts the training and testing subsets from the dataset, and returns a DataLoader ready
	for training a network.
	'''

	subdivisions = inferSubdivisions(dataset.__len__(), split)
	return (torch.utils.data.DataLoader(
		dataset,
		batch_size=batch_size,
		sampler=torch.utils.data.SubsetRandomSampler(list(range(0, subdivisions[0]))),
	),
		torch.utils.data.DataLoader(
		dataset,
		batch_size=batch_size,
		sampler=torch.utils.data.SubsetRandomSampler(list(range(subdivisions[0], subdivisions[1]))),
	))


def getEvaluationDataset(
	dataset: TorchDataset,
	split: tuple[float, float, float] = settings['dataset_split'],
) -> torch.utils.data.DataLoader:
	'''
	Extracts the evaluation subset from the dataset, and returns a DataLoader ready
	for evaluating a network.
	'''

	subdivisions = inferSubdivisions(dataset.__len__(), split)
	return torch.utils.data.DataLoader(
		dataset,
		sampler=torch.utils.data.SubsetRandomSampler(list(range(subdivisions[1], subdivisions[2]))),
	)
