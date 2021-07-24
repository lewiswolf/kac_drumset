# core
import os
import sys

# dependencies
import click					# CLI arguments

# src
sys.path.insert(1, f'{os.getcwd()}/src')
from dataset import generateDataset, loadDataset
from physical_model import PhysicalModel

# test
sys.path.insert(1, f'{os.getcwd()}/test')
from test_utils import TestTone


# set command line flags
@click.command()
@click.option('--train', '-t', is_flag=True, help='Train a new model.')
@click.option('--generate', '-g', is_flag=True, help='Generate targets before training.')
def main(generate: bool, train: bool) -> None:
	# generate a pytorch dataset, or load one if a dataset already exists
	# dataset = generateDataset(PhysicalModel) if generate else loadDataset(PhysicalModel)
	dataset = generateDataset(TestTone) if generate else loadDataset(TestTone)


if __name__ == '__main__':
	main()
