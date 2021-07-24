'''
This file defines an arbitrary audio sample to be used as part of this project. This class
is intended to act as a template for any type of audio sample, as the methods and properties
defined below, which are required for use with this system, will be inherited from this
parent class. This design also ensures that this project's dataset constructor (essentially
dataset.py) can be completely abstracted, and detached from its input.
'''

# core
import math
from typing import TypedDict

# dependencies
import numpy as np				# maths
import numpy.typing as npt		# typing for numpy
import soundfile as sf			# audio read & write

# src
from settings import settings


class SampleMetadata(TypedDict):
	'''
	Metadata format for each audio sample. Each audio sample consists of a wav file stored
	on disk alongside its respective input data (x) and labels (y).
	'''
	filepath: str				# location of .wav file, relative to project directory
	x: list						# input data for the network
	y: list						# labels for each sample


class AudioSample:
	''' Template parent class for an audio sample. '''
	def __init__(self):
		# default variables
		self.sr: int = settings['SAMPLE_RATE']
		self.length: int = math.ceil(settings['DATA_LENGTH'] * self.sr)
		self.metadata: SampleMetadata = {
			'filepath': '',
			'x': [],
			'y': [],
		}

		# call user defined init method
		self.init()

		# generate waveform and metadata
		self.wave: npt.NDArray[np.float64] = self.generateWaveform()

	def init(self) -> None:
		''' template method '''
		pass

	def generateWaveform(self) -> npt.NDArray[np.float64]:
		''' template method '''
		return np.zeros(0)

	def exportWAV(self, absolutePath: str, relativePath: str) -> None:
		'''
		Write the generated waveform to a file.
		params:
			absolutePath: 	The absolute filepath pointing to where the exported file is
							stored. This is used to avoid calling os.getcwd() each time
							this method is called.
			relativePath: 	The relative filepath pointing to the the exported file is
							stored. This is used to populate the metadata.
		'''

		self.metadata['filepath'] = relativePath
		sf.write(absolutePath, self.wave, self.sr)
