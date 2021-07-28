# core
import json
import os
import sys
from typing import cast
import unittest

# dependencies
import pydantic
import torch

# src
sys.path.insert(1, f'{os.getcwd()}/src')
from dataset import DatasetMetadata, parseDataSampleToString, parseMetadataToString


class TestExportedMetadata(unittest.TestCase):
	def test_metadata_stringify(self):
		str = parseMetadataToString()
		for i in range(5):
			str += parseDataSampleToString(i == 4, 'string', torch.zeros(0), torch.zeros(0))
		dict = json.loads(str)
		dict = cast(DatasetMetadata, pydantic.create_model_from_typeddict(DatasetMetadata)(**dict).dict())


if __name__ == '__main__':
	unittest.main()
