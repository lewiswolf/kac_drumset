# core
import os
from typing import Union
from unittest import TestCase

# src
from kac_drumset import AudioSampler
from kac_drumset.utils import clearDirectory


class SamplerTests(TestCase):
	'''
	Tests used in conjunction with `/sampler`.
	'''

	tmp_dir: str = os.path.normpath(f'{os.path.dirname(__file__)}/../tmp')

	def tearDown(self) -> None:
		''' destructor '''
		clearDirectory(self.tmp_dir)

	def test_abstract_sampler(self) -> None:
		'''
		Tests used in conjunction with `sampler/audio_sampler`.
		'''

		class Test(AudioSampler):
			''' The minimum instantiation requirements of AudioSampler. '''
			def __init__(self, duration: float, sr: int) -> None:
				super().__init__(duration, sr)

			def generateWaveform(self) -> None:
				pass

			def getLabels(self) -> list[Union[float, int]]:
				pass

		T = Test(1.0, 48000)
		# This test asserts that the export function exports a wav file.
		test_wav = f'{self.tmp_dir}/test.wav'
		T.export(test_wav)
		self.assertTrue(os.path.exists(test_wav))
