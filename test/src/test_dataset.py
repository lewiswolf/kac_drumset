# core
import os
from unittest import TestCase

# dependencies
import numpy as np 				# maths
import torch					# pytorch

# src
from kac_drumset import InputRepresentation, generateDataset

# test
from kac_drumset import TestSweep
from kac_drumset.utils import clearDirectory, withoutPrinting


class DatasetTests(TestCase):
	'''
	Tests used in conjunction with `/dataset`.
	'''

	tmp_dir: str = os.path.normpath(f'{os.path.dirname(__file__)}/../tmp')
	tone = TestSweep(duration=1.0, sr=48000)

	def tearDown(self) -> None:
		''' destructor '''
		clearDirectory(self.tmp_dir)

	def test_generate_dataset(self) -> None:
		'''
		'''
		with withoutPrinting():
			dataset = generateDataset(
				TestSweep,
				dataset_dir=self.tmp_dir,
				dataset_size=10,
				sampler_settings=TestSweep.Settings({'duration': 1.0, 'sr': 48000}),
			)

		# This test asserts that the dataset is the correct size, both in memory and on disk.
		self.assertEqual(len(dataset.Y), 10)
		self.assertEqual(dataset.__len__(), 10)
		self.assertEqual(len(os.listdir(f'{os.getcwd()}/test/tmp')) - 2, 10)

	def test_IR_end2end(self) -> None:
		IR = InputRepresentation(self.tone.sr, {
			'normalise_input': False,
			'representation_type': 'end2end',
		})
		T = IR.transform(self.tone.waveform)
		# This test asserts that the input waveform and the transform are equivalent.
		self.assertTrue(np.array_equal(self.tone.waveform, T.detach().numpy()))
		# This test asserts that the output tensor is the correct shape and type.
		self.assertEqual(T.shape, IR.transformShape(self.tone.length))
		self.assertEqual(T.dtype, torch.float64)

	def test_IR_fft(self) -> None:
		IR = InputRepresentation(self.tone.sr, {'representation_type': 'fft'})
		spectrogram = IR.transform(self.tone.waveform)
		# This test asserts that the output tensor is the correct shape and type.
		self.assertEqual(spectrogram.shape, IR.transformShape(self.tone.length))
		self.assertEqual(spectrogram.dtype, torch.float64)

	def test_IR_mel(self) -> None:
		# A low n_mels suits the test tone.
		IR = InputRepresentation(self.tone.sr, {'representation_type': 'mel'})
		spectrogram = IR.transform(self.tone.waveform)
		# This test asserts that the output tensor is the correct shape and type.
		self.assertEqual(spectrogram.shape, IR.transformShape(self.tone.length))
		self.assertEqual(spectrogram.dtype, torch.float64)

	def test_IR_normalise(self) -> None:
		# This test asserts that a normalised waveform is always bounded by [-1.0, 1.0].
		norm = InputRepresentation.normalise(self.tone.waveform)
		self.assertEqual(np.max(norm), 1.0)
		self.assertEqual(np.min(norm), -1.0)
