# dependencies
import torch					# pytorch

# src
from dataset import TorchDataset
from settings import settings


def inferSubdivisions() -> list[int]:
	'''
	Calculates the integer splits for the training, testing and validation sets.
	'''

	subdivisions = [round(settings['DATASET_SIZE'] * p) for p in settings['DATASET_SPLIT']]
	# correct errors
	if sum(subdivisions) != settings['DATASET_SIZE']:
		subdivisions[0] += settings['DATASET_SIZE'] - sum(subdivisions)
	# cumulative sums
	s = 0
	for i in range(3):
		s += subdivisions[i]
		subdivisions[i] = s
	return subdivisions


def getTrainingDatasets(dataset: TorchDataset) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
	'''
	Extracts the training and testing subsets from the dataset, and returns a DataLoader ready
	for training a network.
	'''

	subdivisions = inferSubdivisions()
	return (torch.utils.data.DataLoader(
		dataset,
		batch_size=settings['BATCH_SIZE'],
		sampler=torch.utils.data.SubsetRandomSampler(list(range(0, subdivisions[0]))),
	),
		torch.utils.data.DataLoader(
		dataset,
		batch_size=settings['BATCH_SIZE'],
		sampler=torch.utils.data.SubsetRandomSampler(list(range(subdivisions[0], subdivisions[1]))),
	))


def getEvaluationDataset(dataset: TorchDataset) -> torch.utils.data.DataLoader:
	'''
	Extracts the evaluation subset from the dataset, and returns a DataLoader ready
	for evaluating a network.
	'''

	subdivisions = inferSubdivisions()
	return torch.utils.data.DataLoader(
		dataset,
		sampler=torch.utils.data.SubsetRandomSampler(list(range(subdivisions[1], subdivisions[2]))),
	)
