'''
This files contains all of the unit tests applicable to this project. Each class of unit
tests is used in conjunction with a particular project file.
'''

# core
# import json
import os
# import random
# import shutil
import sys
# from typing import cast
import unittest

# dependencies
import cv2					# image processing
import numpy as np 			# maths
# import pydantic 			# runtime type-checking
import torch				# pytorch

# src
sys.path.insert(1, f'{os.getcwd()}/src')
# import dataset as ds
from geometry import isConvex, isColinear, largestVector
from input_features import InputFeatures
# from physical_model import DrumModel, raisedCosine
from random_polygon import RandomPolygon
# from settings import settings

# test
from test_utils import TestSweep, noPrinting
# from test_utils import TestTone


# class DatasetTests(unittest.TestCase):
# '''
# Tests used in conjunction with `dataset.py`.
# '''

# @classmethod
# def tearDownClass(cls) -> None:
# 	'''
# 	Clear the /tmp folder after all tests are done.
# 	Maybe move this...
# 	'''

# 	directory = f'{os.getcwd()}/test/tmp'
# 	for file in os.listdir(directory):
# 		path = f'{directory}/{file}'
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
# 	dataset_dir = f'{os.getcwd()}/test/tmp'
# 	settings_copy = settings.copy()
# 	settings_copy['dataset_size'] = 10
# 	ds.generateDataset(TestTone, dataset_size=10, dataset_dir='test/tmp')

# 	# This test asserts that the initial configuration loads correctly
# 	with open(f'{dataset_dir}/metadata.json') as file:
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
# 	cast(
# 		ds.DatasetMetadata,
# 		pydantic.create_model_from_typeddict(ds.DatasetMetadata)(**JSON).dict(),
# 	)


class GeometryTests(unittest.TestCase):
	'''
	Tests used in conjunction with `geometry.py.
	'''

	def test_properties(self) -> None:
		'''
		Stress test multiple properties of the class RandomPolygon.
		'''

		for i in range(10000):
			polygon = RandomPolygon(20, grid_size=100, allow_concave=True)

			# This test asserts that a polygon has the correct number of vertices.
			self.assertEqual(len(polygon.vertices), polygon.n)

			# This test asserts that the vertices are strictly bounded between 0.0 and 1.0.
			self.assertEqual(np.min(polygon.vertices), 0.0)
			self.assertEqual(np.max(polygon.vertices), 1.0)

			# This test asserts that the shoelaceFunction(), used for calculating the area of
			# a polygon is accurate to at least 6 decimal places. This comparison is bounded
			# due to the shoelaceFunction() being 64-bit, whilst the comparison function,
			# cv2.contourArea(), is 32-bit.
			self.assertAlmostEqual(
				polygon.area,
				cv2.contourArea(polygon.vertices.astype('float32')),
				places=6,
			)

			# This test asserts that no 3 adjacent vertices are colinear.
			for j in range(polygon.n):
				self.assertFalse(isColinear(np.array([
					polygon.vertices[j - 1 if j > 0 else polygon.n - 1],
					polygon.vertices[j],
					polygon.vertices[(j + 1) % polygon.n],
				])))

			if polygon.convex:
				# This test asserts that all supposedly convex polygons are in fact convex.
				# As a result, if this test passes, we can assume that the generateConvex()
				# function works as intended.
				self.assertTrue(isConvex(polygon.vertices))

				# This test asserts that the calculated centroid lies within the polygon. For
				# concave shapes, this test may fail.
				self.assertEqual(polygon.mask[
					round(polygon.centroid[0] * 100),
					round(polygon.centroid[1] * 100),
				], 1)

				# This test asserts that the largest vector is of magnitude 1.0.
				self.assertEqual(largestVector(polygon.vertices)[0], 1.0)


class InputFeatureTests(unittest.TestCase):
	'''
	Tests used in conjunction with `input_features.py`.
	'''

	tone = TestSweep()

	def test_end2end(self) -> None:
		IF = InputFeatures(feature_type='end2end', normalise_input=False)
		T = IF.transform(self.tone.waveform)
		# This test asserts that the input waveform and the transform are equivalent.
		self.assertTrue(np.array_equal(self.tone.waveform, T.detach().numpy()))
		# This test asserts that the output tensor is the correct shape and type.
		self.assertEqual(T.shape, IF.transformShape(self.tone.length))
		self.assertEqual(T.dtype, torch.float64)

	def test_fft(self) -> None:
		IF = InputFeatures(feature_type='fft')
		spectrogram = IF.transform(self.tone.waveform)
		# This test asserts that the output tensor is the correct shape and type.
		self.assertEqual(spectrogram.shape, IF.transformShape(self.tone.length))
		self.assertEqual(spectrogram.dtype, torch.float64)

	def test_mel(self) -> None:
		# A low n_mels suits the test tone.
		IF = InputFeatures(feature_type='mel', n_mels=32)
		spectrogram = IF.transform(self.tone.waveform)
		# This test asserts that the output tensor is the correct shape and type.
		self.assertEqual(spectrogram.shape, IF.transformShape(self.tone.length))
		self.assertEqual(spectrogram.dtype, torch.float64)

	def test_cqt(self) -> None:
		# librosa.cqt() has a number of dependency issues, which clog up the console.
		# There is a warning about n_fft sizes however that should be looked into.
		with noPrinting():
			IF = InputFeatures(feature_type='cqt')
			spectrogram = IF.transform(self.tone.waveform)
		# This test asserts that the output tensor is the correct shape and type.
		self.assertEqual(spectrogram.shape, IF.transformShape(self.tone.length))
		self.assertEqual(spectrogram.dtype, torch.float64)

	def test_normalise(self) -> None:
		# This test asserts that a normalised waveform is always bounded by [-1.0, 1.0].
		norm = InputFeatures.__normalise__(self.tone.waveform)
		self.assertEqual(np.max(norm), 1.0)
		self.assertEqual(np.min(norm), -1.0)


# class PhysicalModelTests(unittest.TestCase):
# 	'''
# 	Tests used in conjunction with `physical_model.py`.
# 	'''

# 	# drum = DrumModel()

# 	def test_CFL_stability(self) -> None:
# 		'''
# 		The Courant number λ = γk/h is used to assert that the CFL stability criterion is upheld.
# 		If λ > 1, the resultant simulation will be unstable.
# 		'''
# 		# For a 1D simulation
# 		# self.assertLessEqual(self.drum.cfl, 1.0)
# 		# For a 2D simulation

# 		drum_size = [0.9, 0.7, 0.5, 0.3, 0.1]
# 		material_density = [0.75, 0.5, 0.25, 0.125, 0.0625]
# 		tension = [3000.0, 2000.0, 1500.0, 1000.0]
# 		for i in range(len(drum_size)):
# 			for j in range(len(material_density)):
# 				for k in range(len(tension)):
# 					drum = DrumModel(
# 						allow_concave=True,
# 						decay_time=1.0,
# 						max_vertices=10,
# 						drum_size=drum_size[i],
# 						material_density=material_density[j],
# 						tension=tension[k],
# 					)
# 					self.assertLessEqual(drum.cfl, 1 / (2 ** 0.5))
# 					drum.length = 1000 # very short simulation
# 					drum.generateWaveform()
# 					self.assertLessEqual(np.max(drum.waveform), 1.0)
# 					self.assertGreaterEqual(np.min(drum.waveform), -1.0)


# 	# def test_energy_conservation(self) -> None:
# 	# 	'''
# 	# 	For an accurate physical simulation, the conservation law of energy must be upheld. This
# 	# 	is both naively tested, using the waveform itself, and by comparing expected bounds on
# 	# 	the Hamiltonian energy throughout the simulation.
# 	# 	'''

# 	# 	# This test asserts that the resultant waveform is always bounded betweenn 1.0 and -1.0.
# 	# 	self.drum.length = 1000 # very short simulation
# 	# 	self.drum.generateWaveform()
# 	# 	self.assertLessEqual(np.max(self.drum.waveform), 1.0)
# 	# 	self.assertGreaterEqual(np.min(self.drum.waveform), -1.0)

# 	def test_raised_cosine(self) -> None:
# 		'''
# 		The raised cosine transform is used as the activation function for a physical model. These
# 		tests assert that the raised cosine works as intended, both in the 1 and 2 dimensional case.
# 		'''

# 		# This test asserts that the one dimensional case has the correct peaks.
# 		rc = raisedCosine((100, ), (50, ), sigma=10)
# 		self.assertEqual(np.max(rc), 1.0)
# 		self.assertEqual(np.min(rc), 0.0)

# 		# This test asserts that the two dimensional case has the correct peaks.
# 		rc = raisedCosine((100, 100), (50, 50), sigma=10)
# 		self.assertEqual(np.max(rc), 1.0)
# 		self.assertEqual(np.min(rc), 0.0)


if __name__ == '__main__':
	unittest.main()
	exit()
