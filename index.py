# core
import os
import sys
from typing import cast, Any

# dependencies
import click					# CLI arguments

# src
sys.path.insert(1, f'{os.getcwd()}/src')
from data_loader import getEvaluationDataset, getTrainingDatasets
from dataset import generateDataset
from nn import trainModel
from physical_model import DrumModel
from settings import settings


# set command line flags
@click.command()
@click.option('--evaluate', '-e', is_flag=True, help='Evaluate a trained model.')
@click.option('--generate', '-g', is_flag=True, help='Generate targets before training.')
@click.option('--train', '-t', is_flag=True, help='Train a new model.')
def main(evaluate: bool, generate: bool, train: bool) -> None:
	# generate a pytorch dataset, or load one if a dataset already exists
	dataset = generateDataset(
		DrumModel,
		sampler_settings=cast(dict[str, Any], settings['pm_settings']),
	)

	# train a new model
	if train:
		trainModel(*getTrainingDatasets(dataset))

	# evaluate a model
	if evaluate:
		evaluationDataset = getEvaluationDataset(dataset)


if __name__ == '__main__':
	main()
	exit()
