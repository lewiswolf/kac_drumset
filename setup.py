# core
import codecs
import os
from setuptools import setup

this = os.path.abspath(os.path.dirname(__file__))
name = 'kac_dataset'
version = '0.0.1'
short_description = 'A dataset generator for arbitrarily shaped drums.'

# import long description from readme.md
with codecs.open(os.path.join(this, 'readme.md'), encoding='utf-8') as rm:
	long_description = '\n' + rm.read()

# import packages from Pipfile
with codecs.open(os.path.join(this, 'Pipfile'), encoding='utf-8') as pf:
	packages = []
	# loop over Pipfile
	p = pf.readlines(1)
	pf.close()
	b = False
	for line in p:
		line = line.replace('\n', '')
		if not line:
			continue
		# find [packages]
		if line[0] == '[':
			if line == '[packages]':
				b = True
				continue
			else:
				b = False
				continue
		# append package names with required version
		if b:
			line = line.split()
			packages.append(f'{line[0]}{line[2][1:-1] if line[2][1:-1] != "*" else ""}')

setup(
	name=name,
	version=version,
	author='Lewis Wolf',
	description=short_description,
	long_description_content_type='text/markdown',
	long_description=long_description,
	packages=['kac_dataset'],
	install_requires=packages,
	package_data={'kac_dataset': ['py.typed']},
	keywords=['kac_dataset'],
	classifiers=[
		'Operating System :: MacOS :: MacOS X',
		'Operating System :: Microsoft :: Windows',
		'Operating System :: Unix',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3 :: Only',
		'Programming Language :: Python :: 3.9',
		'Programming Language :: Python :: 3.10',
		'Typing :: Typed',
	],
)
