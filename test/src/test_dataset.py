# core
# import json
# import os
# import random
# import shutil
# from typing import cast
from unittest import TestCase

# dependencies
# import pydantic

# src
# from kac_dataset import generateDataset


class DatasetTests(TestCase):
	'''
	Tests used in conjunction with `dataset.py`.
	'''

	pass
	# test_dir = {
	# 	abs: f'{os.getcwd()}/test/tmp',
	# 	rel: '/test/tmp',
	# }

	# @classmethod
	# def tearDownClass(cls) -> None:
	# 	'''
	# 	Clear the /tmp folder after all tests are done.
	# 	Maybe move this...
	# 	'''

	# 	for file in os.listdir(self.test_dir.abs):
	# 		path = f'{self.test_dir.abs}/{file}'
	# 		if os.path.isdir(path):
	# 			shutil.rmtree(path)
	# 		elif file != '.gitignore':
	# 			os.remove(path)

	# def test_dataset_settings_confirmation(self) -> None:
	# 	'''
	# 	When a dataset is loaded, the dataset metadata is checked alongside the current
	# 	project settings to look for changes. The tests below are used to confirm that
	# 	these changes are correctly identified.
	# 	'''
	# 	# create an initial dataset with current project settings
	# 	settings_copy = settings.copy()
	# 	settings_copy['dataset_size'] = 10
	# 	generateDataset(TestTone, dataset_size=10, dataset_dir=self.test_dir.rel)

	# 	# This test asserts that the initial configuration loads correctly
	# 	with open(f'{self.test_dir.abs}/metadata.json') as file:
	# 		ds.confirmDatasetSettings(file, {}, s=settings_copy)
	# 		file.close()

	# 	# These tests assert that the DatasetIncompatible error is raised appropriately.
	# 	for i in range(4):
	# 		sampler_settings = {}
	# 		if i == 0:
	# 			settings_copy.update({
	# 				'dataset_size': 11,
	# 			})
	# 		if i == 1:
	# 			settings_copy.update({
	# 				'data_length': 0.0,
	# 				'dataset_size': 10,
	# 			})
	# 		if i == 2:
	# 			settings_copy.update({
	# 				'data_length': settings['data_length'],
	# 				'sample_rate': 0,
	# 			})
	# 		if i == 3:
	# 			settings_copy.update({
	# 				'sample_rate': settings['sample_rate'],
	# 			})
	# 			sampler_settings = {'some_key': 'some_val'}

	# 		with open(f'{dataset_dir}/metadata.json') as file:
	# 			self.assertRaises(
	# 				ds.DatasetIncompatible,
	# 				ds.confirmDatasetSettings,
	# 				file,
	# 				sampler_settings,
	# 				s=settings_copy,
	# 			)
	# 			file.close()

	# 	# These tests assert that the InputIncompatible error is raised appropriately.
	# 	# After raising this error, the file is also checked to be at the appropriate
	# 	# read location.
	# 	for i in range(1):
	# 		if i == 0:
	# 			settings_copy.update({
	# 				'input_features': 'fft' if settings['input_features'] == 'end2end' else 'end2end',
	# 			})

	# 		with open(f'{dataset_dir}/metadata.json') as file:
	# 			self.assertRaises(
	# 				ds.InputIncompatible,
	# 				ds.confirmDatasetSettings,
	# 				file,
	# 				{},
	# 				s=settings_copy,
	# 			)
	# 			self.assertEqual(file.readlines(1), ['{\n'])
	# 			file.close()

	# def test_generated_dataset(self) -> None:
	# 	'''
	# 	Tests associated with generating a dataset. These test check for the correct size
	# 	and data type of the dataset, both in memory and on disk.
	# 	'''

	# 	# Test with y.
	# 	with noPrinting():
	# 		dataset = ds.generateDataset(TestTone, dataset_size=10, dataset_dir='test/tmp')

	# 	# This test asserts that the dataset is the correct size, both in memory and on disk.
	# 	self.assertEqual(dataset.__len__(), 10)
	# 	self.assertEqual(len(dataset.Y), 10)
	# 	self.assertEqual(len(os.listdir(f'{os.getcwd()}/test/tmp')) - 2, 10)

	# 	# This test asserts that x and y are the correct data types.
	# 	for i in range(10):
	# 		x, y = dataset.__getitem__(i)
	# 		self.assertEqual(x.dtype, torch.float64)
	# 		self.assertNotEqual(y, None)
	# 		if y: # for mypy only
	# 			self.assertEqual(y.dtype, torch.float64)

	# 	# Test without y.
	# 	with noPrinting():
	# 		dataset = ds.generateDataset(TestSweep, dataset_size=10, dataset_dir='test/tmp')

	# 	# This test asserts that dataset.Y does not exist.
	# 	self.assertFalse(hasattr(dataset, 'Y'))

	# 	# This test asserts that x and y are the correct data types.
	# 	for i in range(10):
	# 		x, y = dataset.__getitem__(i)
	# 		self.assertEqual(x.dtype, torch.float64)
	# 		self.assertEqual(y, None)

	# def test_metadata_stringify(self) -> None:
	# 	'''
	# 	First stringifies the dataset's metadata, ready for exporting a .json file, and
	# 	then checks that it containes the correct values and data types.
	# 	'''

	# 	# number of tests
	# 	n = 10

	# 	# Test with y labels.
	# 	str = ds.parseMetadataToString(
	# 		sampler_settings={
	# 			'key1': 'value1',
	# 			'key2': 'value2',
	# 		},
	# 	)
	# 	for i in range(n):
	# 		str += ds.parseDataSampleToString({'filepath': '', 'x': [], 'y': [1]}, i == n - 1)
	# 	JSON = json.loads(str)
	# 	# This test asserts that the y labels exist.
	# 	for i in range(n):
	# 		self.assertTrue('y' in JSON['data'][i])
	# 	# This test asserts that the dataset matches the type specification.
	# 	# self.assertTrue(isinstance()) is not used here as TypedDicts do not
	# 	# support instance/class checks.
	# 	cast(
	# 		ds.DatasetMetadata,
	# 		pydantic.create_model_from_typeddict(ds.DatasetMetadata)(**JSON).dict(),
	# 	)

	# 	# Test with falsey y labels.
	# 	str = ds.parseMetadataToString(
	# 		sampler_settings={
	# 			'key1': 'value1',
	# 			'key2': 'value2',
	# 		},
	# 	)
	# 	for i in range(n):
	# 		if random.getrandbits(1):
	# 			str += ds.parseDataSampleToString({'filepath': '', 'x': []}, i == n - 1)
	# 		else:
	# 			str += ds.parseDataSampleToString({'filepath': '', 'x': [], 'y': []}, i == n - 1)
	# 	JSON = json.loads(str)
	# 	# This test asserts that the y labels do not exist.
	# 	for i in range(n):
	# 		self.assertFalse('y' in JSON['data'][i])
	# 	# This test asserts that the dataset matches the type specification.
	# 	# self.assertTrue(isinstance()) is not used here as TypedDicts do not
	# 	# support instance/class checks.
	# 	cast(
	# 		ds.DatasetMetadata,
	# 		pydantic.create_model_from_typeddict(ds.DatasetMetadata)(**JSON).dict(),
	# 	)
