# core
import os
import sys

# dependencies
import click					# CLI arguments
import torch					# pytorch

# src
sys.path.insert(1, f'{os.getcwd()}/src')
from data_loader import getEvaluationDataset, getTrainingDatasets
from dataset import generateDataset, loadDataset
from nn import trainModel
from physical_model import PhysicalModel


# set command line flags
@click.command()
@click.option('--evaluate', '-e', is_flag=True, help='Evaluate a trained model.')
@click.option('--generate', '-g', is_flag=True, help='Generate targets before training.')
@click.option('--train', '-t', is_flag=True, help='Train a new model.')
def main(evaluate: bool, generate: bool, train: bool) -> None:
	# necessary to enforce dtype throughout the project, see todo.md ->
	# 'Internal types for nested lists, numpy arrays and pytroch tensors'
	torch.set_default_dtype(torch.float64)

	# generate a pytorch dataset, or load one if a dataset already exists
	dataset = generateDataset(PhysicalModel) if generate else loadDataset(PhysicalModel)

	# train a new model
	if train:
		trainModel(*getTrainingDatasets(dataset))

	# evaluate a model
	# if evaluate:
	# 	evaluationDataset = getEvaluationDataset(dataset)


if __name__ == '__main__':
	exit(main())
