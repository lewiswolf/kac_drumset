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
		'''
		First stringifies the dataset's metadata, ready for exporting a .json file, and
		then checks that it containes the correct data types.
		'''

		str = parseMetadataToString()
		for i in range(5):
			str += parseDataSampleToString(i == 4, 'string', torch.zeros(0), torch.zeros(0))
		cast(DatasetMetadata, pydantic.create_model_from_typeddict(DatasetMetadata)(**json.loads(str)).dict())


if __name__ == '__main__':
	unittest.main()
