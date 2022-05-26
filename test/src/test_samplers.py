# core
import os
from typing import Union
from unittest import TestCase

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

		# This test asserts that The Courant number λ = γk/h, which is used to confirm that the CFL stability criterion is
		# upheld. If λ > 1 / (dimensionality)^0.5, the resultant simulation will be unstable.
		self.assertLessEqual(model.cfl, 1 / (2 ** 0.5))
