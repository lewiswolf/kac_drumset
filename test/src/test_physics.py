# core
from unittest import TestCase

# dependencies
import numpy as np 			# maths

# src
from kac_drumset.physics import (
	calculateCircularAmplitudes,
	calculateCircularSeries,
	calculateRectangularAmplitudes,
	FDTD_2D,
	FDTDWaveform2D,
	raisedCosine,
)


class PhysicsTests(TestCase):
	'''
	Tests used in conjunction with `/physics`.
	'''

	def test_bessel(self) -> None:
		'''
		Tests used in conjunction with circular_modes.hpp.
		'''

		# This test asserts that the amplitude calculation is programmed correctly.
		series = calculateCircularSeries(10, 10)
		for r in [1., -1.]:
			for theta in [0., np.pi / 2, np.pi, np.pi * 2]:
				self.assertAlmostEqual(
					float(np.max(calculateCircularAmplitudes(r, theta, series))),
					0.,
					places=15,
				)

	def test_fdtd(self) -> None:
		'''
		Tests used in conjunction with `fdtd.hpp`.
		'''

		# matrices
		u_0 = np.zeros((10, 10))
		u_1 = np.pad(raisedCosine((8, 8), (3, 3)), 1, mode='constant')
		B = np.pad(np.ones((8, 8), dtype=np.int8), 1, mode='constant')
		# courant number and decay coefficients
		cfl = 1 / (2 ** 0.5)
		c_0 = cfl ** 2
		c_1 = 2 * (1 - 2 * (cfl ** 2))
		c_2 = 1.

		# Test iterator with a square simulation
		for u in FDTD_2D(u_0=u_0.tolist(), u_1=u_1.tolist(), B=B.tolist(), c_0=c_0, c_1=c_1, c_2=c_2, T=20):
			# This test asserts that the conservation law of energy is upheld. This is here naively tested, using the waveform
			# itself, but should also be confirmed by comparing expected bounds on the Hamiltonian energy throughout the
			# simulation.
			self.assertFalse(np.isnan(u).any())
			self.assertLessEqual(np.max(u), 1.)
			self.assertGreaterEqual(np.min(u), -1.)

		# Test waveform generator with a square simulation
		waveform = FDTDWaveform2D(u_0=u_0, u_1=u_1, B=B, c_0=c_0, c_1=c_1, c_2=c_2, T=10, w=(4, 4))
		# This test asserts that the conservation law of energy is upheld. This is here naively tested, using the waveform
		# itself, but should also be confirmed by comparing expected bounds on the Hamiltonian energy throughout the
		# simulation.
		self.assertFalse(np.isnan(waveform).any())
		self.assertLessEqual(np.max(waveform), 1.)
		self.assertGreaterEqual(np.min(waveform), -1.)

	def test_poisson(self) -> None:
		'''
		Tests used in conjunction with rectangular_modes.hpp.
		'''

		# This test asserts that the amplitude calculation is programmed correctly.
		for e in [1., 1.5, 2.]:
			e_root = e ** 0.5
			e_inv = 1 / (e ** 0.5)
			self.assertAlmostEqual(
				float(np.max(calculateRectangularAmplitudes((0., 0.), 10, 10, e))),
				0.,
				places=28,
			)
			self.assertAlmostEqual(
				float(np.max(calculateRectangularAmplitudes((e_root, 0.), 10, 10, e))),
				0.,
				places=28,
			)
			self.assertAlmostEqual(
				float(np.max(calculateRectangularAmplitudes((0., e_inv), 10, 10, e))),
				0.,
				places=28,
			)
			self.assertAlmostEqual(
				float(np.max(calculateRectangularAmplitudes((e_root, e_inv), 10, 10, e))),
				0.,
				places=28,
			)

	def test_raised_cosine(self) -> None:
		'''
		The raised cosine transform is used as the activation function for a physical model. These tests assert that the
		raised cosine works as intended, both in the 1 and 2 dimensional cases.
		'''

		# This test asserts that the one dimensional case has the correct peaks.
		rc = raisedCosine((100, ), (50, ), sigma=10)
		self.assertEqual(rc[50], 1.)
		self.assertEqual(np.max(rc), 1.)
		self.assertEqual(np.min(rc), 0.)
		self.assertGreater(rc[49], 0.)
		self.assertGreater(rc[51], 0.)

		# This test asserts that the two dimensional case has the correct peaks.
		rc = raisedCosine((100, 100), (50, 50), sigma=10)
		self.assertEqual(rc[50, 50], 1.)
		self.assertEqual(np.max(rc), 1.)
		self.assertEqual(np.min(rc), 0.)
		self.assertGreater(rc[49, 50], 0.)
		self.assertGreater(rc[51, 50], 0.)
		self.assertGreater(rc[50, 49], 0.)
		self.assertGreater(rc[50, 51], 0.)
