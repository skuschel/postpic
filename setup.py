#!/usr/bin/python

import os
from setuptools import setup
from epochsdftools import __version__


setup(name='epochsdftools',
    version=__version__,
	author='Stephan Kuschel',
	author_email='stephan.kuschel@uni-jena.de',
	description='Toolkit um Daten aus PIC Simulationen zu verarbeiten und darzustellen.',
	packages=['epochsdftools'],
	install_requires=['matplotlib', 'numpy', 'scipy']
	)
