# core
import os
import sys

# dependencies
import click										# CLI arguments
import pydantic										# runtime type-checking

# src
sys.path.insert(1, os.path.join(os.getcwd(), 'src'))
from settings import settings, Settings				# creates a project settings object
from dataset import generateDataset, loadDataset	# methods for handling and generating a dataset

# test
sys.path.insert(1, os.path.join(os.getcwd(), 'test'))
from test_utils import withProfiler


# add the CUDA SDK to the environment variables
if settings['PATH_2_CUDA'] is not None and 'CUDA_HOME' not in os.environ:
	os.environ['CUDA_HOME'] = settings['PATH_2_CUDA']


# set command line flags
@click.command()
@click.option('--train', '-t', is_flag=True, help='Train a new model.')
@click.option('--generate', '-g', is_flag=True, help='Generate targets before training.')
def main(generate: bool, train: bool) -> None:
	# type check settings object at runtime
	# TO FIX: see todo.md -> 'pydantic.create_model_from_typeddict has an incompatible type error'
	try:
		pydantic.create_model_from_typeddict(Settings)(**settings)
	except pydantic.ValidationError as e:
		print(f"ERROR: The project settings are not configured correctly. {e}")
		sys.exit()

	# generate a pytorch dataset, or load one if a dataset already exists
	dataset = generateDataset() if generate else loadDataset()
	# withProfiler(generateDataset if generate else loadDataset, 5)


if __name__ == '__main__':
	main()
