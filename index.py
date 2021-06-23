# core libraries
import os
import sys

# dependencies
import click			# CLI arguments
from tqdm import tqdm	# CLI progress bar

# src files
sys.path.insert(1, os.path.join(os.getcwd(), 'src'))
from settings import settings	# creates a settings object


# set command line flags
@click.command()
@click.option('--train', '-t', is_flag=True, help='Train a new model.')
@click.option('--generate', '-g', is_flag=True, help='Generate targets before training.')
def main(generate: bool, train: bool) -> None:
	if (generate):
		# generate dataset
		print('generate dataset')
	if (train):
		# train model
		print('train model')

	# # progress bar example for later
	# with tqdm(
	# 	bar_format='{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}, {elapsed} < {remaining}, {rate_fmt}  ',
	# 	unit=' units',
	# 	total=settings['test']
	# ) as pbar:
	# 	for i in range(settings['test']):
	# 		# do stuff
	# 		pbar.update(1)


if __name__ == '__main__':
	main()
