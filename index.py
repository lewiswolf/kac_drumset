# core
import os
import sys

# dependencies
import click										# CLI arguments

# src
sys.path.insert(1, os.path.join(os.getcwd(), 'src'))
from dataset import generateDataset, loadDataset	# methods for handling and generating a dataset


# set command line flags
@click.command()
@click.option('--train', '-t', is_flag=True, help='Train a new model.')
@click.option('--generate', '-g', is_flag=True, help='Generate targets before training.')
def main(generate: bool, train: bool) -> None:
	# generate a dataset, or load one if one already exists
	datasetJSON = generateDataset() if generate else loadDataset()


if __name__ == '__main__':
	main()
