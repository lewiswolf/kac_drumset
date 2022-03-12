'''
Utilities to be used when testing the package locally.
'''

# core
import contextlib
import cProfile
import os
import pstats
import sys
from typing import Any, Callable, Iterator


@contextlib.contextmanager
def noPrinting(allow_errors: bool = False) -> Iterator[Any]:
	'''
	This wrapper can used around blocks of code to silence calls to print(), as well as
	optionally silence error messages.
	'''

	with open(os.devnull, 'w') as dummy_file:
		if not allow_errors:
			sys.stderr = dummy_file
		sys.stdout = dummy_file
		yield
		dummy_file.close()
	sys.stderr = sys.__stderr__
	sys.stdout = sys.__stdout__


def withProfiler(func: Callable, n: int, *args: Any, **kwargs: Any) -> None:
	'''
	Calls the input function using cProfile to generate a performance report in the console.
	Prints the n most costly functions.
	'''

	with cProfile.Profile() as pr:
		func(*args, **kwargs)
	stats = pstats.Stats(pr)
	stats.sort_stats(pstats.SortKey.TIME)
	stats.print_stats(n)
