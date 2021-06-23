# core libraries
import os
import sys
import time

# dependencies
import click			# CLI arguments
from tqdm import tqdm	# CLI progress bar

# src files
sys.path.insert(1, os.path.join(os.getcwd(), 'src'))
from settings import settings


def main():
	with tqdm(
		bar_format='{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}, {elapsed} < {remaining}, {rate_fmt}  ',
		unit=' units',
		total=settings['test']
	) as pbar:
		for i in range(settings['test']):
			time.sleep(0.6)
			pbar.update(1)


if __name__ == '__main__':
	main()
