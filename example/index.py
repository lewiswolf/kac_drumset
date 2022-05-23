from kac_drumset import (
	RepresentationSettings,	# typing for the representation_settings
	TestTone,				# an AudioSampler used for generating raw waveforms.
	TorchDataset,			# the dataset class.
	generateDataset,
	loadDataset,
	transformDataset,
)


def dataset() -> None:
	'''
	This example demonstrates all of the methods used to generate, load, and modify a dataset.
	'''

	# Generating a dataset takes as its first argument an AudioSampler, each of which has its own customised settings
	# constructor. To configure the representation settings, an dict of type RepresentationSettings is passed to the
	# function. This function also takes an absolute path as an argument to store the dataset.
	representation_settings: RepresentationSettings = {'output_type': 'end2end'}
	dataset: TorchDataset = generateDataset(
		TestTone,
		dataset_size=10,
		representation_settings=representation_settings,
		sampler_settings=TestTone.Settings({
			'duration': 1.0,
			'waveshape': 'sin',
			'sample_rate': 48000,
		}),
	)
	# Datasets can be loaded using the method below, which takes one optional argument, the dataset directory, which points
	# the method to the dataset.
	dataset = loadDataset()
	# To redefine the input representation, the below method is used. This modifies the metadata for the dataset, such that
	# this method is only executed when the current settings are different from the ones passed to the function.
	representation_settings = {'output_type': 'fft'}
	dataset = transformDataset(dataset, representation_settings)


if __name__ == '__main__':
	dataset()
	exit()
