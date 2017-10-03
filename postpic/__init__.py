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

"""
+--------------+
|   POSTPIC    |
+--------------+

The open source particle-in-cell post processor.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from . import helper
from . import datahandling
from . import particles
from .datahandling import *
from .helper import *
from .particles import *
from . import datareader
from .datareader import chooseCode, readDump, readSim
from . import plotting
from ._version import get_versions

__all__ = ['helper']
__all__ += datahandling.__all__
__all__ += helper.__all__
__all__ += particles.__all__
__all__ += ['datareader', 'plotting']
# high level functions
__all__ += ['chooseCode', 'readDump', 'readSim']

__version__ = get_versions()['version']
__git_version__ = get_versions()['full-revisionid']
del get_versions
