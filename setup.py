#!/usr/bin/env python2

from setuptools import setup
from epochsdftools import __version__


setup(name='epochsdftools',
    version=__version__,
	author='Stephan Kuschel',
	author_email='stephan.kuschel@gmail.de',
	description='Provides easy-to-use functions to create customized plots from simulations data created by the epoch code. This toolset can be easily adapted to also support the output of other particle in cell code.',
	packages=['epochsdftools'],
	install_requires=['matplotlib', 'numpy', 'scipy']
	)
