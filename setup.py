#!/usr/bin/env python2
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
from setuptools import setup

setup(name='postpic',
      version='0.1.1',
      author='Stephan Kuschel',
      author_email='stephan.kuschel@gmail.de',
      description='The open source particle-in-cell post processor.',
      url='http://github.com/skuschel/postpic',
      packages=['postpic'],
      license='GPLv3+',
      install_requires=['matplotlib', 'numpy>=1.7', 'scipy'],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Scientific/Engineering :: Visualization',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2.7',
          'Operating System :: Unix',
          'Operating System :: MacOS',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)']
      )
