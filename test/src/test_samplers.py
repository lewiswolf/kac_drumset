# core
import os
from typing import Union
from unittest import TestCase

# dependencies
import numpy as np 			# maths

# src
from kac_drumset import AudioSampler, FDTDModel, SamplerSettings
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
			def __init__(self, duration: float = 1.0, sample_rate: int = 48000) -> None:
				super().__init__(duration, sample_rate)

			def generateWaveform(self) -> None:
				pass

			def getLabels(self) -> list[Union[float, int]]:
				pass

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

	def test_fdtd_model(self) -> None:
		'''
		Tests used in conjunction with `samplers/fdtd_model.py`.
		'''

		# This test asserts that model correctly mounts with both its minimum requirements and type safety.
		settings: FDTDModel.Settings = {'duration': 1.0, 'sample_rate': 48000}
		model = FDTDModel(**settings)

		# This test asserts that the labels default to an empty array when no waveform has been generated.
		self.assertEqual(model.getLabels(), [])

		# generate a random shape and dirichlet boundary conditions.
		model.updateProperties()

		# This test asserts that a shape was properly defined after updating the model's properties.
		self.assertTrue(hasattr(model, 'shape'))
		# This test asserts that a boolean mask was properly defined after updating the model's properties.
		self.assertEqual(model.B[model.strike], 1.0)
		# This test asserts that the model returns the vertices of the shape as its labels.
		self.assertEqual(len(model.getLabels()), model.max_vertices)
		self.assertEqual(
			model.getLabels()[:model.shape.n],
			model.shape.vertices.tolist(),
		)

		# generate a distribution of drums to assert that the sampler works with various configurations
		drum_size = [0.9, 0.7, 0.5, 0.3, 0.1]
		material_density = [0.75, 0.5, 0.25, 0.125, 0.0625]
		tension = [3000.0, 2000.0, 1500.0, 1000.0]
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
					self.assertLessEqual(np.max(model.waveform), 1.0)
					self.assertGreaterEqual(np.min(model.waveform), -1.0)
