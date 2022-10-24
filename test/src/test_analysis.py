
# core
from unittest import TestCase

# src
from kac_drumset import TestTone
from kac_drumset.analysis import dominantModes


class AnalysisTests(TestCase):
	'''
	Tests used in conjunction with `/analysis`.
	'''

	def test_spectrum(self) -> None:
		'''
		Test spectral analyses.
		'''

		f_0 = 440.
		T = TestTone(
			duration=1.,
			f_0=440.,
			sample_rate=48000,
			waveshape='saw',
		)
		T.generateWaveform()

		# this test asserts that the 0th mode is correct
		pred_f_0 = dominantModes(T.waveform, T.sample_rate, fft_size=48000)[0]
		self.assertGreater(pred_f_0, f_0 - 1.)
		self.assertLess(pred_f_0, f_0 + 1.)
