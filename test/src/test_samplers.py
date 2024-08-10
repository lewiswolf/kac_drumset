# core
import os
from unittest import TestCase

# dependencies
import numpy as np 			# maths

# src
from kac_drumset.geometry import (
	Circle,
	# ConvexPolygon,
	Ellipse,
	# IrregularStar,
	Shape,
	# TravellingSalesmanPolygon,
)
from kac_drumset.samplers import (
	BesselModel,
	FDTDModel,
	LaméModel,
	PoissonModel,
)
from kac_prediction.utils import clearDirectory


class SamplerTests(TestCase):
	'''
	Tests used in conjunction with `/samplers`.
	'''

	tmp_dir: str = os.path.normpath(f'{os.path.dirname(__file__)}/../tmp')

	def tearDown(self) -> None:
		''' destructor '''
		clearDirectory(self.tmp_dir)

	def test_bessel_model(self) -> None:
		'''
		Tests used in conjunction with `samplers/bessel_model.py`.
		'''

		# This test asserts that model correctly mounts with both its minimum requirements and type safety.
		settings: BesselModel.Settings = {'duration': 1., 'sample_rate': 48000, 'decay_time': np.inf}
		model = BesselModel(**settings)

		# This test asserts that the labels default to an empty array when no waveform has been generated.
		self.assertEqual(model.getLabels(), {})

		# This test asserts that decay_time: np.inf works as expected.
		self.assertEqual(model.decay, 0.)

		# stress test the bessel model
		for i in range(100):
			model.updateProperties(i)

			# This test asserts that a size and strike location were properly defined after updating
			# the model's properties.
			self.assertTrue(hasattr(model, 'F'))
			self.assertTrue(hasattr(model, 'L'))
			self.assertTrue(hasattr(model, 'strike'))

			# This test asserts that the 2D series has correct shape
			self.assertEqual(model.F.shape, (10, 10))

			# This test asserts that the model returns a drum_size.
			self.assertEqual(len(model.getLabels()['drum_size']), 1)

			# This test asserts that the model returns a valid polar strike location.
			self.assertEqual(len(model.getLabels()['strike_location']), 2)
			self.assertGreaterEqual(model.getLabels()['strike_location'][0], -1.)
			self.assertLessEqual(model.getLabels()['strike_location'][0], 1.)
			self.assertGreaterEqual(model.getLabels()['strike_location'][1], 0.)
			self.assertLessEqual(model.getLabels()['strike_location'][1], np.pi)

			# This test asserts that the waveform is not distorted.
			model.generateWaveform()
			self.assertLessEqual(model.waveform.max(), 1.)
			self.assertGreaterEqual(model.waveform.min(), -1.)

	def test_fdtd_model(self) -> None:
		'''
		Tests used in conjunction with `samplers/fdtd_model.py`.
		'''

		# test using all shapes
		shapes: list[type[Shape]] = [Circle, Ellipse]
		for shape in shapes:

			# This test asserts that model correctly mounts with both its minimum requirements and type safety.
			settings: FDTDModel.Settings = {
				'arbitrary_shape': shape,
				'decay_time': np.inf,
				'duration': 1.,
				'sample_rate': 48000,
			}
			model = FDTDModel(**settings)

			# This test asserts that the labels default to an empty array when no waveform has been generated.
			self.assertEqual(model.getLabels(), {})

			# This test asserts that decay_time: np.inf works as expected.
			self.assertEqual(model.c_2, 1.)

			# generate a random shape and dirichlet boundary conditions.
			settings = {'arbitrary_shape': shape, 'duration': 1., 'sample_rate': 48000}
			model = FDTDModel(**settings)
			model.updateProperties()

			# This test asserts that a shape was properly defined after updating the model's properties.
			self.assertTrue(hasattr(model, 'shape'))

			# This test asserts that the model returns the sample_location and the strike_location as its labels.
			self.assertLessEqual(len(model.getLabels()['sample_location']), 2)
			self.assertEqual(len(model.getLabels()['strike_location']), 2)

			# generate a distribution of drums to assert that the sampler works with various configurations
			drum_sizes = [0.9, 0.7, 0.5, 0.3, 0.1]
			material_densities = [0.75, 0.5, 0.25, 0.125, 0.0625]
			tensions = [3000., 2000., 1500., 1000.]
			for drum_size in drum_sizes:
				for material_density in material_densities:
					for tension in tensions:
						model = FDTDModel(
							arbitrary_shape=shape,
							drum_size=drum_size,
							duration=0.02,
							material_density=material_density,
							sample_rate=48000,
							tension=tension,
						)
						model.updateProperties()

						# This test asserts that the boundary considitions matches the size H
						# model.H + 2 is used to account for model.B being padded
						self.assertEqual(model.B.shape[0], model.H + 2)

						# This test asserts that the listening location is always within the drum.
						# model.H + 1 is used to account for model.B being padded
						self.assertTrue(model.shape.isPointInside(model.w))

						# test both the centroid strike location and the randomised shape location.
						for i in range(2):
							model.updateProperties(i)

							# This test asserts that the strike location is always within the drum.
							self.assertTrue(model.shape.isPointInside(model.strike))

							# This test asserts that The Courant number λ = γk/h, which is used to confirm that the
							# CFL stability criterion is upheld. If λ > 1 / (dimensionality)^0.5, the resultant
							# simulation will be unstable.
							self.assertLessEqual(model.cfl, 1 / (2 ** 0.5))

							# This test asserts that the conservation law of energy is upheld. This is here naively
							# tested, using the waveform itself, but should also be confirmed by comparing expected
							# bounds on the Hamiltonian energy throughout the simulation.
							model.generateWaveform()
							self.assertNotEqual(np.sum(model.waveform), 0.)
							self.assertFalse(np.isnan(model.waveform).any())
							self.assertLessEqual(model.waveform.max(), 1.)
							self.assertGreaterEqual(model.waveform.min(), -1.)

	def test_lamé_model(self) -> None:
		'''
		Tests used in conjunction with `samplers/lamé_model.py`.
		'''

		# This test asserts that model correctly mounts with both its minimum requirements and type safety.
		settings: LaméModel.Settings = {'duration': 1., 'sample_rate': 48000, 'decay_time': np.inf}
		model = LaméModel(**settings)

		# This test asserts that the labels default to an empty array when no waveform has been generated.
		self.assertEqual(model.getLabels(), {})

		# This test asserts that decay_time: np.inf works as expected.
		self.assertEqual(model.decay, 0.)

		# stress test the bessel model
		for i in range(100):
			model.updateProperties(i)

			# This test asserts that a size and strike location were properly defined after updating
			# the model's properties.
			self.assertTrue(hasattr(model, 'F'))
			self.assertTrue(hasattr(model, 'L'))
			self.assertTrue(hasattr(model, 'strike'))

			# This test asserts that the 2D series has correct shape
			self.assertEqual(model.F.shape, (10, 10))

			# This test asserts that the model returns a drum_size.
			self.assertEqual(len(model.getLabels()['drum_size']), 1)

			# This test asserts that the waveform is not distorted.
			model.generateWaveform()
			self.assertLessEqual(model.waveform.max(), 1.)
			self.assertGreaterEqual(model.waveform.min(), -1.)

	def test_poisson_model(self) -> None:
		'''
		Tests used in conjunction with `samplers/poisson_model.py`.
		'''

		# This test asserts that model correctly mounts with both its minimum requirements and type safety.
		settings: PoissonModel.Settings = {'duration': 1., 'sample_rate': 48000, 'decay_time': np.inf}
		model = PoissonModel(**settings)

		# This test asserts that the labels default to an empty array when no waveform has been generated.
		self.assertEqual(model.getLabels(), {})

		# This test asserts that decay_time: np.inf works as expected.
		self.assertEqual(model.decay, 0.)

		# stress test the poisson model
		for i in range(100):
			model.updateProperties(i)
			# This test asserts that a size, aspect ratio and strike location were properly defined after
			# updating the model's properties.
			self.assertTrue(hasattr(model, 'epsilon'))
			self.assertTrue(hasattr(model, 'F'))
			self.assertTrue(hasattr(model, 'L'))
			self.assertTrue(hasattr(model, 'strike'))

			# This test asserts that the 2D series has correct shape
			self.assertEqual(model.F.shape, (10, 10))

			# This test asserts that the model returns a drum_size.
			self.assertEqual(len(model.getLabels()['aspect_ratio']), 1)
			self.assertEqual(len(model.getLabels()['drum_size']), 1)

			# This test asserts that the model returns a valid cartesian strike location.
			# The strike location should be normalised such that {x [0, 1]} => {x [0, (Ɛ^0.5)]} &
			# {y, [0, 1]} => {y, [0,  1 / (Ɛ^0.5)]}
			self.assertEqual(len(model.getLabels()['strike_location']), 2)
			self.assertGreaterEqual(model.getLabels()['strike_location'][0], 0.)
			self.assertLessEqual(model.getLabels()['strike_location'][0], 1.)
			self.assertGreaterEqual(model.getLabels()['strike_location'][1], 0.)
			self.assertLessEqual(model.getLabels()['strike_location'][1], 1.)

			# This test asserts that the waveform is not distorted.
			model.generateWaveform()
			self.assertLessEqual(model.waveform.max(), 1.)
			self.assertGreaterEqual(model.waveform.min(), -1.)
