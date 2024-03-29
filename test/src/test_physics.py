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
	raisedTriangle,
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
					float(calculateCircularAmplitudes(r, theta, series).max()),
					0.,
					places=15,
				)

	def test_fdtd(self) -> None:
		'''
		Tests used in conjunction with `fdtd.hpp`.
		'''

		# matrices
		u_0 = np.zeros((10, 10))
		u_1 = np.pad(raisedCosine((8, 8), (3., 3.)), 1, mode='constant')
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
			self.assertLessEqual(u.max(), 1.)
			self.assertGreaterEqual(u.min(), -1.)

		# Test waveform generator with a square simulation
		waveform = FDTDWaveform2D(u_0=u_0, u_1=u_1, B=B, c_0=c_0, c_1=c_1, c_2=c_2, T=20, w=(4, 4))
		# This test asserts that the conservation law of energy is upheld. This is here naively tested, using the waveform
		# itself, but should also be confirmed by comparing expected bounds on the Hamiltonian energy throughout the
		# simulation.
		self.assertFalse(np.isnan(waveform).any())
		self.assertLessEqual(waveform.max(), 1.)
		self.assertGreaterEqual(waveform.min(), -1.)

	def test_poisson(self) -> None:
		'''
		Tests used in conjunction with rectangular_modes.hpp.
		'''

		# This test asserts that the amplitude calculation is programmed correctly.
		for e in [1., 1.5, 2.]:
			e_root = e ** 0.5
			e_inv = 1 / (e ** 0.5)
			self.assertAlmostEqual(
				float(calculateRectangularAmplitudes((0., 0.), 10, 10, e).max()),
				0.,
				places=28,
			)
			self.assertAlmostEqual(
				float(calculateRectangularAmplitudes((e_root, 0.), 10, 10, e).max()),
				0.,
				places=28,
			)
			self.assertAlmostEqual(
				float(calculateRectangularAmplitudes((0., e_inv), 10, 10, e).max()),
				0.,
				places=28,
			)
			self.assertAlmostEqual(
				float(calculateRectangularAmplitudes((e_root, e_inv), 10, 10, e).max()),
				0.,
				places=28,
			)

	def test_initial_conditions(self) -> None:
		'''
		Tests used in conjunction with `initial_conditions.hpp`.
		'''

		# This test asserts that the one dimensional raised cosine has the correct peaks.
		rc = raisedCosine((100, ), (50., ), sigma=10)
		self.assertEqual(rc[50], 1.)
		self.assertEqual(rc.max(), 1.)
		self.assertEqual(rc.min(), 0.)
		self.assertGreater(rc[49], 0.)
		self.assertGreater(rc[51], 0.)

		# This test asserts that the two dimensional raised cosine has the correct peaks.
		rc = raisedCosine((100, 100), (50., 50.), sigma=10)
		self.assertEqual(rc[50, 50], 1.)
		self.assertEqual(rc.max(), 1.)
		self.assertEqual(rc.min(), 0.)
		self.assertGreater(rc[49, 50], 0.)
		self.assertGreater(rc[51, 50], 0.)
		self.assertGreater(rc[50, 49], 0.)
		self.assertGreater(rc[50, 51], 0.)

		# This test asserts that the one dimensional triangular distribution has the correct peaks.
		t = raisedTriangle((100, ), (50., ), x_ab=(30., 70.))
		self.assertEqual(t[50], 1.)
		self.assertEqual(t.max(), 1.)
		self.assertEqual(t.min(), 0.)
		self.assertGreater(t[49], 0.)
		self.assertGreater(t[51], 0.)

		# This test asserts that the two dimensional triangular distribution has the correct peaks.
		t = raisedTriangle((100, 100), (50., 50), x_ab=(30., 70.), y_ab=(30., 70.))
		self.assertEqual(rc[50, 50], 1.)
		self.assertEqual(t.max(), 1.)
		self.assertEqual(t.min(), 0.)
		self.assertGreater(rc[49, 50], 0.)
		self.assertGreater(rc[51, 50], 0.)
		self.assertGreater(rc[50, 49], 0.)
		self.assertGreater(rc[50, 51], 0.)
