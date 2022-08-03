# core
from unittest import TestCase

# dependencies
import numpy as np 			# maths

# src
from kac_drumset.physics import FDTDWaveform2D, raisedCosine


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
		rc = raisedCosine((100, ), (50, ), sigma=10)
		self.assertEqual(rc[50], 1.0)
		self.assertEqual(np.max(rc), 1.0)
		self.assertEqual(np.min(rc), 0.0)

		# This test asserts that the two dimensional case has the correct peaks.
		rc = raisedCosine((100, 100), (50, 50), sigma=10)
		self.assertEqual(rc[50, 50], 1.0)
		self.assertEqual(np.max(rc), 1.0)
		self.assertEqual(np.min(rc), 0.0)

	def test_fdtd(self) -> None:
		'''
		Tests used in conjunction with `fdtd.hpp`.
		'''

		# courant number
		cfl = 1 / (2 ** 0.5)
		# square simulation
		waveform = FDTDWaveform2D(
			u_0=np.zeros((10, 10)),
			u_1=np.pad(raisedCosine((8, 8), (3, 3)), 1, mode='constant'),
			B=np.pad(np.ones((8, 8), dtype=np.int8), 1, mode='constant'),
			c_0=cfl ** 2,
			c_1=2 * (1 - 2 * (cfl ** 2)),
			c_2=1.,
			T=10,
			w=(4, 4),
		)

		# This test asserts that the conservation law of energy is upheld. This is here naively tested, using the waveform
		# itself, but should also be confirmed by comparing expected bounds on the Hamiltonian energy throughout the
		# simulation.
		self.assertFalse(np.isnan(waveform).any())
		self.assertLessEqual(np.max(waveform), 1.0)
		self.assertGreaterEqual(np.min(waveform), -1.0)
