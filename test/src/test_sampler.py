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
			def __init__(self, duration: float = 1.0, sr: int = 48000) -> None:
				super().__init__(duration, sr)

			def generateWaveform(self) -> None:
				pass

			def getLabels(self) -> list[Union[float, int]]:
				pass

		# This test asserts that the export function exports a wav file.
		test_wav = f'{self.tmp_dir}/test.wav'
		Test().export(test_wav)
		self.assertTrue(os.path.exists(test_wav))

		for i in [8000, 16000, 22050, 44100, 48000, 88200, 96000]:
			# This test asserts that the wav file will export all necessary sample rates and bit depths.
			bit_16 = f'{self.tmp_dir}/test-{16}-{i}.wav'
			Test(sr=i).export(bit_16, bit_depth=16)
			self.assertTrue(os.path.exists(bit_16))
			bit_24 = f'{self.tmp_dir}/test-{24}-{i}.wav'
			Test(sr=i).export(bit_24, bit_depth=24)
			self.assertTrue(os.path.exists(bit_24))
			bit_32 = f'{self.tmp_dir}/test-{32}-{i}.wav'
			Test(sr=i).export(bit_32, bit_depth=32)
			self.assertTrue(os.path.exists(bit_32))
