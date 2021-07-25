# core
import os
import sys

# dependencies
import click					# CLI arguments
import torch					# pytorch

# src
sys.path.insert(1, f'{os.getcwd()}/src')
from dataset import generateDataset, loadDataset
from physical_model import PhysicalModel


# set command line flags
@click.command()
@click.option('--train', '-t', is_flag=True, help='Train a new model.')
@click.option('--generate', '-g', is_flag=True, help='Generate targets before training.')
def main(generate: bool, train: bool) -> None:
	# necessary to enforce dtype throughout the project, see todo.md ->
	# 'Internal types for nested lists, numpy arrays and pytroch tensors'
	torch.set_default_dtype(torch.float64)

	# generate a pytorch dataset, or load one if a dataset already exists
	dataset = generateDataset(PhysicalModel) if generate else loadDataset(PhysicalModel)


if __name__ == '__main__':
	main()
