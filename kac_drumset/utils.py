'''
This file contains utility functions used throughout the main codebase. These functions are intended for either
debugging the package or interfacing with the file system/command line.
'''

# core
import contextlib
import cProfile
import os
import pstats
import re
import shutil
import sys
import time
from typing import Any, Callable, Iterator

__all__ = [
	'clearDirectory',
	'printEmojis',
	'withoutPrinting',
	'withProfiler',
	'withTimer',
]


def clearDirectory(absolutePath: str, excludes: list[str] = ['.gitignore']) -> None:
	'''
	Completely clears all files and folders from the input directory.
	params:
		absolutePath	absolute filepath to the directory for clearing
		excludes		files or subdirectories to exclude from clearing
	'''

	for file in os.listdir(absolutePath):
		if file in excludes:
			continue
		path = f'{absolutePath}/{file}'
		shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)


def printEmojis(s: str) -> None:
	'''
	Checks whether or not the operating system is mac or linux. If so, emojis are printed as normal, else they are
	filtered from the string.
	'''

	if sys.platform in ['linux', 'darwin']:
		print(s)
	else:
		regex = re.compile(
			'['
			u'\U00002600-\U000026FF' # miscellaneous
			u'\U00002700-\U000027BF' # dingbats
			u'\U0001F1E0-\U0001F1FF' # flags (iOS)
			u'\U0001F600-\U0001F64F' # emoticons
			u'\U0001F300-\U0001F5FF' # symbols & pictographs I
			u'\U0001F680-\U0001F6FF' # transport & map symbols
			u'\U0001F900-\U0001F9FF' # symbols & pictographs II
			u'\U0001FA70-\U0001FAFF' # symbols & pictographs III
			']+',
			flags=re.UNICODE,
		)
		print(regex.sub(r'', s).strip())


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


def withProfiler(func: Callable, n: int, *args: Any, **kwargs: Any) -> None:
	'''
	Calls the input function using cProfile to generate a performance report in the console. Prints the n most costly
	functions.
	'''

	with cProfile.Profile() as pr:
		func(*args, **kwargs)
	stats = pstats.Stats(pr)
	stats.sort_stats(pstats.SortKey.TIME)
	stats.print_stats(n)


def withTimer(func: Callable, *args: Any, **kwargs: Any) -> None:
	'''
	Calls the input function and posts its runtime to the console.
	'''

	t1 = time.time_ns()
	func(*args, **kwargs)
	print(f'{(time.time_ns() - t1) / (10 ** 9):.10f}')
