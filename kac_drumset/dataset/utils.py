'''
This file contains variables and methods reused throughout this module.
'''

# core
from typing import Union

# dependencies
import torch					# pytorch

__all__ = [
	'tqdm_settings',
]


# settings for the tqdm progress bar
tqdm_settings = {
	'bar_format': '{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}, Elapsed: {elapsed}, ETA: {remaining}, {rate_fmt}  ',
	'unit': ' data samples',
}


def listToTensor(d: dict[str, list[Union[float, int]]]) -> dict[str, torch.Tensor]:
	t = {}
	for k, v in d.items():
		t[k] = torch.as_tensor(v)
	return t


def tensorToList(t: dict[str, torch.Tensor]) -> dict[str, list[Union[float, int]]]:
	d = {}
	for k, v in t.items():
		d[k] = v.tolist()
	return d
