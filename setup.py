#!/usr/bin/python

import os
from setuptools import setup



setup(name='epochsdftools',
    version='0.1'
	author='Stephan Kuschel',
	author_email='stephan.kuschel@uni-jena.de',
	description='Toolkit um Daten aus PIC Simulationen zu verarbeiten und darzustellen.',
	packages=['epochsdftools','epochsdftools.v1p0', 'epochsdftools.v1p1'],
	install_requires=['matplotlib']
	)
