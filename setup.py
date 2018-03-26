#!/usr/bin/env python

# Copyright (C) 2016 Stephan Kuschel
#
# This file is part of postpic.
#
# postpic is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# postpic is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with postpic. If not, see <http://www.gnu.org/licenses/>.
#
from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy
import os

import versioneer

setup(name='postpic',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      include_package_data = True,
      author='Stephan Kuschel',
      author_email='stephan.kuschel@gmail.de',
      description='The open source particle-in-cell post processor.',
      url='https://github.com/skuschel/postpic',
      packages=find_packages(include=['postpic*']),
      ext_modules = cythonize("postpic/particles/_particlestogrid.pyx"),
      include_dirs = [numpy.get_include()],
      license='GPLv3+',
      setup_requires=['cython>=0.18', 'numpy>=1.8'],
      install_requires=['matplotlib>=1.3',
                        # ndarray.tobytes was introduced in np 1.9 and workaround in vtk routines
                        # does not work for python 2
                        'numpy>=1.8', 'numpy>=1.9;python_version<"3.0"',
                        'scipy', 'future', 'urllib3', 'numexpr',
                        'cython>=0.18', 'functools32;python_version<"3.0"'],
      extras_require = {
        'h5 reader for openPMD support':  ['h5py'],
        'sdf support for EPOCH reader':  ['sdf'],
        'PyPNG read png files': ['pypng'],
        'Pillow to read other image files': ['pillow']},
      keywords = ['PIC', 'particle-in-cell', 'plasma', 'physics', 'plasma physics',
                  'laser', 'laser plasma', 'particle acceleration'],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Scientific/Engineering :: Visualization',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Operating System :: Unix',
          'Operating System :: MacOS',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)']
      )
