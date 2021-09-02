'''
This files contains all of the unit tests applicable to this project. Each class of unit
tests is used in conjunction with a particular project file.
'''

# core
import json
import os
import sys
from typing import cast
import unittest

# dependencies
import cv2					# image processing
import numpy as np 			# maths
import pydantic 			# runtime type-checking
import torch				# pytorch

# src
sys.path.insert(1, f'{os.getcwd()}/src')
from dataset import DatasetMetadata, parseDataSampleToString, parseMetadataToString
from geometry import RandomPolygon
from physical_model import PhysicalModel


class DatasetTests(unittest.TestCase):
	'''
	Tests used in conjunction with `dataset.py`.
	'''

	def test_metadata_stringify(self) -> None:
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

	def test_properties(self) -> None:
		'''
		Test multiple properties of the class RandomPolygon
		'''

		for i in range(100):
			polygon = RandomPolygon(20, 200)

			# This test asserts that the vertices are strictly bounded between 0.0 and 1.0.
			self.assertEqual(np.min(polygon.vertices), 0.0)
			self.assertEqual(np.max(polygon.vertices), 1.0)

			# This test asserts that the shoelaceFunction(), used for calculating the area of
			# a polygo,n is accurate to at least 7 decimal places. This comparison is bounded
			# due to shoelaceFunction() being 64-bit, and the comparison function
			# cv2.contourArea() being 32-bit.
			self.assertAlmostEqual(
				polygon.area,
				cv2.contourArea(polygon.vertices.astype('float32')),
				places=7,
			)

			# This test asserts that the calculated centroid lies within the polygon.
			self.assertEqual(polygon.mask[
				round(polygon.centroid[0] * 200),
				round(polygon.centroid[1] * 200),
			], 1)


class PhysicalModelTests(unittest.TestCase):
	'''
	Tests used in conjunction with `physical_model.py`.
	'''

	drum = PhysicalModel()

	def test_CFL_stability(self) -> None:
		'''
		The Courant number λ = γk/h is used to assert that the CFL stability criterion is upheld.
		If λ > 1, the resultant simulation will be unstable.
		'''

		self.assertLessEqual(self.drum.cfl, 1.0)
		# Special cases such as λ = sqrt of something...


if __name__ == '__main__':
	exit(unittest.main())
