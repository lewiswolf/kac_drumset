'''
Custom build script used to import this package's metadata from both the readme and Pipfile.
'''

# core
import codecs
import os
from setuptools import find_namespace_packages
from skbuild import setup


this = os.path.abspath(os.path.dirname(__file__))
name = 'kac_drumset'
version = '2.0.0'
short_description = 'Analysis tools and a dataset generator for arbitrarily shaped drums.'

# import long description from readme.md
with codecs.open(os.path.join(this, 'readme.md'), encoding='utf-8') as readme:
	long_description = '\n' + readme.read()

setup(
	author='Lewis Wolstanholme',
	author_email='lewiswolstanholme@gmail.com',
	classifiers=[
		'Operating System :: MacOS :: MacOS X',
		'Operating System :: Microsoft :: Windows',
		'Operating System :: Unix',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3 :: Only',
		'Programming Language :: Python :: 3.11',
		'Programming Language :: Python :: 3.12',
		'Programming Language :: Python :: 3.13',
		'Typing :: Typed',
	],
	cmake_install_dir=f'{os.path.relpath(os.path.dirname(__file__), os.getcwd())}/kac_drumset/externals',
	description=short_description,
	long_description=long_description,
	include_package_data=True,
	install_requires=['kac_prediction', 'numpy>=2.3', 'opencv-python>=4.11'],
	keywords=['kac_drumset'],
	long_description_content_type='text/markdown',
	name=name,
	packages=find_namespace_packages(),
	package_data={'kac_drumset': ['py.typed']},
	version=version,
)
