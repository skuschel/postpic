#!/usr/bin/env python2

from setuptools import setup
from postpic import __version__


setup(name='postpic',
      version=__version__,
      author='Stephan Kuschel',
      author_email='stephan.kuschel@gmail.de',
      description='The open source particle-in-cell post processor.',
      url='http://github.com/skuschel/postpic',
      packages=['postpic'],
      install_requires=['matplotlib', 'numpy', 'scipy'],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Scientific/Engineering :: Visualization']
      )
