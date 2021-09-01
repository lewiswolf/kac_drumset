# core
import json
import os
import sys
from typing import cast
import unittest

# dependencies
import numpy as np
import pydantic
import torch

# src
sys.path.insert(1, f'{os.getcwd()}/src')
from dataset import DatasetMetadata, parseDataSampleToString, parseMetadataToString
from geometry import RandomPolygon
from physical_model import PhysicalModel


class DatasetTests(unittest.TestCase):
	'''
	Tests used in conjunction with `dataset.py`.
	'''

	def test_metadata_stringify(self):
		'''
		First stringifies the dataset's metadata, ready for exporting a .json file, and
		then checks that it containes the correct data types.
		'''
		str = parseMetadataToString()
		for i in range(5):
			str += parseDataSampleToString('filepath', torch.zeros(0), torch.zeros(0), i == 4)
		cast(DatasetMetadata, pydantic.create_model_from_typeddict(DatasetMetadata)(**json.loads(str)).dict())


class GeometryTests(unittest.TestCase):
	'''
	Tests used in conjunction with `geometry.py.
	'''

	def test_floating_point_error(self):
		'''
		Asserts that the minimum and maximum value of the vertices are between 0.0 and 1.0
		respectively.
		'''
		for i in range(100):
			shape = RandomPolygon(20, 200)
			self.assertEqual(np.min(shape.vertices), 0.0)
			self.assertEqual(np.max(shape.vertices), 1.0)


class PhysicalModelTests(unittest.TestCase):
	'''
	Tests used in conjunction with `physical_model.py`.
	'''

	drum = PhysicalModel()

	def test_CFL_stability(self):
		'''
		The courante number λ = γk/h assures the CFL stability criterion. If λ > 1,
		the simulation will be unstable.
		'''
		self.assertLessEqual(self.drum.cfl, 1.0)
		# Special cases such as λ = sqrt of something...


if __name__ == '__main__':
	unittest.main()
