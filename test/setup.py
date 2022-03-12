'''
Custom build script for local testing library.
'''

# core
from setuptools import setup

name = 'test_utils'

setup(
	author='Lewis Wolf',
	install_requires=[],
	name=name,
	packages=['test_utils'],
	package_data={'test_utils': ['py.typed']},
)
