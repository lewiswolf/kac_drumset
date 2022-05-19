# core
import os

# src
from .types import SampleMetadata


def parseDataSampleToString(s: SampleMetadata, finalLine: bool) -> str:
	'''
	Parse a data sample, as defined by SampleMetadata, to a raw JSON string with line breaks.
	This function is designed to be implemented within a for loop.
	'''

	str = r'{'
	str += fr'{os.linesep}"filepath": "{s["filepath"]}",'
	str += fr'{os.linesep}"x": {s["x"]}'
	str += fr',{os.linesep}"y": {s["y"]}'
	str += f'{os.linesep}'
	str += r'}]}' if finalLine else r'},' + f'{os.linesep}'
	return str
