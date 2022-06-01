'''
This file contains variables reused throughout this module.
'''

__all__ = [
	'tqdm_settings',
]


# settings for the tqdm progress bar
tqdm_settings = {
	'bar_format': '{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}, Elapsed: {elapsed}, ETA: {remaining}, {rate_fmt}  ',
	'unit': ' data samples',
}
