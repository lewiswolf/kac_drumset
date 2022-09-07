# core
import os


def dataset() -> None:
	'''
	This example demonstrates all of the methods used to generate, load, and modify a dataset.
	'''

	from kac_drumset import (
		# methods
		generateDataset,
		loadDataset,
		transformDataset,
		# classes
		FDTDModel,
		# types
		RepresentationSettings,	# typing for the representation_settings
		TorchDataset,			# the dataset class
	)

	# Generating a dataset takes as its first argument an AudioSampler, each of which has its own customised settings
	# constructor. To configure the representation settings, a dict of type RepresentationSettings is passed to the
	# function. This function also takes an absolute path as an argument to store the dataset.
	dataset_dir: str = os.path.normpath(f'{os.path.dirname(__file__)}/data')
	representation_settings: RepresentationSettings = {'output_type': 'end2end'}
	dataset: TorchDataset = generateDataset(
		FDTDModel,
		dataset_dir=dataset_dir,
		dataset_size=10,
		representation_settings=representation_settings,
		sampler_settings=FDTDModel.Settings({
			'duration': 1.0,
			'sample_rate': 48000,
		}),
	)
	# Datasets can be loaded using the method below, which takes only the dataset directory as its argument.
	dataset = loadDataset(dataset_dir)
	# To redefine the input representation, the below method is used. This modifies the metadata for the dataset, such that
	# this method is only executed when the current settings are different from the ones passed to the function.
	representation_settings = {'output_type': 'fft'}
	dataset = transformDataset(dataset, representation_settings)


if __name__ == '__main__':
	dataset()
	exit()
