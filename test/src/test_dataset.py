# core
from unittest import TestCase

# dependencies
import numpy as np
import torch

# src
from kac_drumset.dataset import InputFeatures

# test
from kac_drumset import TestSweep
from kac_drumset.utils import withoutPrinting


class DatasetTests(TestCase):
	'''
	Tests used in conjunction with `/dataset`.
	'''

	tone = TestSweep()

	def test_end2end(self) -> None:
		IF = InputFeatures(feature_type='end2end', normalise_input=False, sr=self.tone.sr)
		T = IF.transform(self.tone.waveform)
		# This test asserts that the input waveform and the transform are equivalent.
		self.assertTrue(np.array_equal(self.tone.waveform, T.detach().numpy()))
		# This test asserts that the output tensor is the correct shape and type.
		self.assertEqual(T.shape, IF.transformShape(self.tone.length))
		self.assertEqual(T.dtype, torch.float64)

	def test_fft(self) -> None:
		IF = InputFeatures(feature_type='fft', sr=self.tone.sr)
		spectrogram = IF.transform(self.tone.waveform)
		# This test asserts that the output tensor is the correct shape and type.
		self.assertEqual(spectrogram.shape, IF.transformShape(self.tone.length))
		self.assertEqual(spectrogram.dtype, torch.float64)

	def test_mel(self) -> None:
		# A low n_mels suits the test tone.
		IF = InputFeatures(feature_type='mel', spectrogram_settings={'n_mels': 32}, sr=self.tone.sr)
		spectrogram = IF.transform(self.tone.waveform)
		# This test asserts that the output tensor is the correct shape and type.
		self.assertEqual(spectrogram.shape, IF.transformShape(self.tone.length))
		self.assertEqual(spectrogram.dtype, torch.float64)

	def test_cqt(self) -> None:
		# librosa.cqt() has a number of dependency issues, which clog up the console.
		# There is a warning about n_fft sizes however that should be looked into.
		with withoutPrinting():
			IF = InputFeatures(feature_type='cqt', sr=self.tone.sr)
			spectrogram = IF.transform(self.tone.waveform)
		# This test asserts that the output tensor is the correct shape and type.
		self.assertEqual(spectrogram.shape, IF.transformShape(self.tone.length))
		self.assertEqual(spectrogram.dtype, torch.float64)

	def test_normalise(self) -> None:
		# This test asserts that a normalised waveform is always bounded by [-1.0, 1.0].
		norm = InputFeatures.__normalise__(self.tone.waveform)
		self.assertEqual(np.max(norm), 1.0)
		self.assertEqual(np.min(norm), -1.0)
