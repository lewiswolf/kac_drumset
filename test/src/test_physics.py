# core
from unittest import TestCase

# dependencies
import numpy as np 			# maths

# src
from kac_drumset.physics import raisedCosine


class PhysicsTests(TestCase):
	'''
	Tests used in conjunction with `/physics`.
	'''

	def test_raised_cosine(self) -> None:
		'''
		The raised cosine transform is used as the activation function for a physical model. These tests assert that the
		raised cosine works as intended, both in the 1 and 2 dimensional cases.
		'''

		# This test asserts that the one dimensional case has the correct peaks.
		rc = raisedCosine((50, ), (100, ), sigma=10)
		self.assertEqual(rc[50], 1.0)
		self.assertEqual(np.max(rc), 1.0)
		self.assertEqual(np.min(rc), 0.0)

		# This test asserts that the two dimensional case has the correct peaks.
		rc = raisedCosine((50, 50), (100, 100), sigma=10)
		self.assertEqual(rc[50, 50], 1.0)
		self.assertEqual(np.max(rc), 1.0)
		self.assertEqual(np.min(rc), 0.0)
