'''
This file contains variables and methods reused throughout this module.
'''

# core
from typing import Any

# dependencies
import torch					# pytorch

__all__ = [
	# methods
	'classLocalsToKwargs',
	'listToTensor',
	'tensorToList',
	# variables
	'tqdm_settings',
]


# settings for the tqdm progress bar
tqdm_settings = {
	'bar_format': '{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}, Elapsed: {elapsed}, ETA: {remaining}, {rate_fmt}  ',
	'unit': ' data samples',
}


def classLocalsToKwargs(d: dict[str, Any]) -> dict[str, Any]:
	'''
	This methods removes unwanted (key, value) pairs from the result of locals() called within a class.
	'''
	return {key: value for key, value in d.items() if key not in ['self', '__class__']}


def listToTensor(d: dict[str, list[float | int]]) -> dict[str, torch.Tensor]:
	''' Convert a dictionary of lists to a dictionary of tensors. '''
	t = {}
	for k, v in d.items():
		t[k] = torch.as_tensor(v)
	return t


def tensorToList(t: dict[str, torch.Tensor]) -> dict[str, list[float | int]]:
	''' Convert a dictionary of tensors to a dictionary of lists. '''
	d = {}
	for k, v in t.items():
		d[k] = v.tolist()
	return d
