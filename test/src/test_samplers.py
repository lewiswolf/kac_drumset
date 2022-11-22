# core
import os
from typing import Union
from unittest import TestCase

# dependencies
import numpy as np 			# maths

# src
from kac_drumset import (
	AudioSampler,
	BesselModel,
	FDTDModel,
	PoissonModel,
	SamplerSettings,
)
from kac_drumset.utils import clearDirectory


class SamplerTests(TestCase):
	'''
	Tests used in conjunction with `/samplers`.
	'''

	tmp_dir: str = os.path.normpath(f'{os.path.dirname(__file__)}/../tmp')

	def tearDown(self) -> None:
		''' destructor '''
		clearDirectory(self.tmp_dir)

	def test_abstract_sampler(self) -> None:
		'''
		Tests used in conjunction with `dataset/audio_sampler`.
		'''

		class Test(AudioSampler):
			''' The minimum instantiation requirements of AudioSampler. '''
			def __init__(self, duration: float = 1., sample_rate: int = 48000) -> None:
				super().__init__(duration, sample_rate)

			def generateWaveform(self) -> None:
				pass

			def getLabels(self) -> dict[str, list[Union[float, int]]]:
				return {}

			def updateProperties(self, i: Union[int, None] = None) -> None:
				pass

			class Settings(SamplerSettings):
				pass

		# This test asserts that the export function exports a wav file.
		test_wav = f'{self.tmp_dir}/test.wav'
		Test().export(test_wav)
		self.assertTrue(os.path.exists(test_wav))

		for sr in [8000, 16000, 22050, 44100, 48000, 88200, 96000]:
			# This test asserts that the wav file will export all necessary sample rates and bit depths.
			bit_16 = f'{self.tmp_dir}/test-{16}-{sr}.wav'
			Test(sample_rate=sr).export(bit_16, bit_depth=16)
			self.assertTrue(os.path.exists(bit_16))
			bit_24 = f'{self.tmp_dir}/test-{24}-{sr}.wav'
			Test(sample_rate=sr).export(bit_24, bit_depth=24)
			self.assertTrue(os.path.exists(bit_24))
			bit_32 = f'{self.tmp_dir}/test-{32}-{sr}.wav'
			Test(sample_rate=sr).export(bit_32, bit_depth=32)
			self.assertTrue(os.path.exists(bit_32))

	def test_bessel(self) -> None:
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

		# This test asserts that the 2D series has correct shape
		self.assertEqual(model.series.shape, (10, 10))

		# stress test the bessel model
		for i in range(1000):
			model.updateProperties(i)
			# This test asserts that a size and strike location were properly defined after updating
			# the model's properties.
			self.assertTrue(hasattr(model, 'L'))
			self.assertTrue(hasattr(model, 'strike'))
			# This test asserts that the model returns a drum_size.
			self.assertEqual(len(model.getLabels()['drum_size']), 1)
			# This test asserts that the model returns a valid polar strike location.
			self.assertEqual(len(model.getLabels()['strike_location']), 2)
			self.assertGreaterEqual(model.getLabels()['strike_location'][0], -1.)
			self.assertLessEqual(model.getLabels()['strike_location'][0], 1.)
			self.assertGreaterEqual(model.getLabels()['strike_location'][1], 0.)
			self.assertLessEqual(model.getLabels()['strike_location'][1], np.pi)

	def test_fdtd_model(self) -> None:
		'''
		Tests used in conjunction with `samplers/fdtd_model.py`.
		'''

		# This test asserts that model correctly mounts with both its minimum requirements and type safety.
		settings: FDTDModel.Settings = {'duration': 1., 'sample_rate': 48000, 'decay_time': np.inf}
		model = FDTDModel(**settings)

		# This test asserts that the labels default to an empty array when no waveform has been generated.
		self.assertEqual(model.getLabels(), {})

		# This test asserts that decay_time: np.inf works as expected.
		self.assertEqual(model.c_2, 1.)

		# generate a random shape and dirichlet boundary conditions.
		settings = {'duration': 1., 'sample_rate': 48000}
		model = FDTDModel(**settings)
		model.updateProperties()

		# This test asserts that a shape was properly defined after updating the model's properties.
		self.assertTrue(hasattr(model, 'shape'))
		# This test asserts that a boolean mask was properly defined after updating the model's properties.
		self.assertEqual(model.B[model.strike], 1.)
		# This test asserts that the model returns the vertices and the strike location as its labels.
		self.assertEqual(len(model.getLabels()['strike_location']), 2)
		self.assertLessEqual(len(model.getLabels()['vertices']), model.max_vertices)

		# generate a distribution of drums to assert that the sampler works with various configurations
		drum_size = [0.9, 0.7, 0.5, 0.3, 0.1]
		material_density = [0.75, 0.5, 0.25, 0.125, 0.0625]
		tension = [3000., 2000., 1500., 1000.]
		for i in range(len(drum_size)):
			for j in range(len(material_density)):
				for k in range(len(tension)):
					model = FDTDModel(
						duration=0.02,
						sample_rate=48000,
						drum_size=drum_size[i],
						material_density=material_density[j],
						tension=tension[k],
					)
					model.updateProperties()
					model.generateWaveform()

					# This test asserts that The Courant number λ = γk/h, which is used to confirm that the
					# CFL stability criterion is upheld. If λ > 1 / (dimensionality)^0.5, the resultant
					# simulation will be unstable.
					self.assertLessEqual(model.cfl, 1 / (2 ** 0.5))

					# This test asserts that the conservation law of energy is upheld. This is here naively
					# tested, using the waveform itself, but should also be confirmed by comparing expected
					# bounds on the Hamiltonian energy throughout the simulation.
					self.assertFalse(np.isnan(model.waveform).any())
					self.assertLessEqual(np.max(model.waveform), 1.)
					self.assertGreaterEqual(np.min(model.waveform), -1.)

	def test_poisson(self) -> None:
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
		for i in range(1000):
			model.updateProperties(i)
			# This test asserts that a size, aspect ratio and strike location were properly defined after
			# updating the model's properties.
			self.assertTrue(hasattr(model, 'epsilon'))
			self.assertTrue(hasattr(model, 'L'))
			self.assertTrue(hasattr(model, 'series'))
			self.assertTrue(hasattr(model, 'strike'))
			# This test asserts that the 2D series has correct shape
			self.assertEqual(model.series.shape, (10, 10))
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
