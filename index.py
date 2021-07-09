# core
import os
import sys

# dependencies
import click										# CLI arguments

# src
sys.path.insert(1, os.path.join(os.getcwd(), 'src'))
from dataset import generateDataset, loadDataset	# methods for handling and generating a dataset

# test
sys.path.insert(1, os.path.join(os.getcwd(), 'test'))
from test_utils import withProfiler


# set command line flags
@click.command()
@click.option('--train', '-t', is_flag=True, help='Train a new model.')
@click.option('--generate', '-g', is_flag=True, help='Generate targets before training.')
def main(generate: bool, train: bool) -> None:
	# generate a pytorch dataset, or load one if a dataset already exists
	dataset = generateDataset() if generate else loadDataset()
	# withProfiler(generateDataset if generate else loadDataset, 5)


if __name__ == '__main__':
	main()
