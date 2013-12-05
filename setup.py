#!/usr/bin/python

import os
from setuptools import setup

version=0.1

setup(name='epochsdftools',
	author='Stephan Kuschel',
	packages=['epochsdftools','epochsdftools.v1p1'],
	install_requires=['matplotlib']
	)
