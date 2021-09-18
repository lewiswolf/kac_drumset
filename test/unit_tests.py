'''
This files contains all of the unit tests applicable to this project. Each class of unit
tests is used in conjunction with a particular project file.
'''

# core
import os
import sys
import unittest

# dependencies
import cv2					# image processing
import numpy as np 			# maths
import torch				# pytorch

# src
sys.path.insert(1, f'{os.getcwd()}/src')
from geometry import RandomPolygon, isConvex, isColinear
from input_features import InputFeatures

# test
from test_utils import TestSweep, noPrinting


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

			# This test asserts that the vertices are strictly bounded between 0.0 and 1.0.
			self.assertEqual(np.min(polygon.vertices), 0.0)
			self.assertEqual(np.max(polygon.vertices), 1.0)

			# This test asserts that the shoelaceFunction(), used for calculating the area of
			# a polygon is accurate to at least 7 decimal places. This comparison is bounded
			# due to the shoelaceFunction() being 64-bit, and the comparison function,
			# cv2.contourArea(), being 32-bit.
			self.assertAlmostEqual(
				polygon.area,
				cv2.contourArea(polygon.vertices.astype('float32')),
				places=7,
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
				self.assertTrue(isConvex(polygon.n, polygon.vertices))
				# This test asserts that the calculated centroid lies within the polygon. For
				# concave shapes, this test may fail.
				self.assertEqual(polygon.mask[
					round(polygon.centroid[0] * 100),
					round(polygon.centroid[1] * 100),
				], 1)


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
		self.assertEqual(torch.float64, T.dtype)

	def test_fft(self) -> None:
		IF = InputFeatures(feature_type='fft')
		spectrogram = IF.transform(self.tone.waveform)
		# This test asserts that the output tensor is the correct shape and type.
		self.assertEqual(spectrogram.shape, IF.transformShape(self.tone.length))
		self.assertEqual(torch.float64, spectrogram.dtype)

	def test_mel(self) -> None:
		# A low n_mels suits the test tone.
		IF = InputFeatures(feature_type='mel', n_mels=32)
		spectrogram = IF.transform(self.tone.waveform)
		# This test asserts that the output tensor is the correct shape and type.
		self.assertEqual(spectrogram.shape, IF.transformShape(self.tone.length))
		self.assertEqual(torch.float64, spectrogram.dtype)

	def test_cqt(self) -> None:
		# librosa.cqt() has a number of dependency issues, which clog up the console.
		# There is a warning about n_fft sizes however that should be looked into.
		with noPrinting():
			IF = InputFeatures(feature_type='cqt')
			spectrogram = IF.transform(self.tone.waveform)
		# This test asserts that the output tensor is the correct shape and type.
		self.assertEqual(spectrogram.shape, IF.transformShape(self.tone.length))
		self.assertEqual(torch.float64, spectrogram.dtype)

	def test_normalise(self) -> None:
		# This test asserts that a normalised waveform is always bounded by [-1.0, 1.0].
		self.assertEqual(np.max(InputFeatures.__normalise__(self.tone.waveform)), 1.0)
		self.assertEqual(np.min(InputFeatures.__normalise__(self.tone.waveform)), -1.0)


if __name__ == '__main__':
	exit(unittest.main())
