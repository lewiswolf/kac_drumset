'''
This file contains utility functions used throughout the main codebase. These functions are intended for either
debugging the package or interfacing with the file system/command line.
'''

# core
import contextlib
import cProfile
import os
import pstats
import sys
import time
from typing import Any, Callable, Iterator

__all__ = [
	'withoutPrinting',
	'withProfiler',
	'withTimer',
]


@contextlib.contextmanager
def withoutPrinting(allow_errors: bool = False) -> Iterator[Any]:
	'''
	This wrapper is used around blocks of code to silence calls to print(), as well as optionally silence error messages.
	'''

	with open(os.devnull, 'w') as dummy_file:
		if not allow_errors:
			sys.stderr = dummy_file
		sys.stdout = dummy_file
		yield
		dummy_file.close()
	sys.stderr = sys.__stderr__
	sys.stdout = sys.__stdout__


def withProfiler(func: Callable[..., Any], n: int, *args: Any, **kwargs: Any) -> None:
	'''
	Calls the input function using cProfile to generate a performance report in the console. Prints the n most costly
	functions.
	'''

	with cProfile.Profile() as pr:
		func(*args, **kwargs)
	stats = pstats.Stats(pr)
	stats.sort_stats(pstats.SortKey.TIME)
	stats.print_stats(n)


def withTimer(func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
	'''
	Calls the input function and posts its runtime to the console.
	'''

	t1 = time.time_ns()
	func(*args, **kwargs)
	print(f'{(time.time_ns() - t1) / (10 ** 9):.10f}')
