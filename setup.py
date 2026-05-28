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


setup(
        cmdclass=versioneer.get_cmdclass(),
        ext_modules = cythonize("postpic/particles/_particlestogrid.pyx"),
        include_dirs = [numpy.get_include()],
        include_package_data = True
)
      
