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
from .datahandling import *
from .helper import PhysicalConstants
from .particles import *
from . import datareader
from .datareader import chooseCode, readDump, readSim
from . import plotting

__all__ = ['helper']
__all__ += datahandling.__all__
__all__ += ['PhysicalConstants']
__all__ += particles.__all__
__all__ += ['datareader', 'plotting']
# high level functions
__all__ += ['chooseCode', 'readDump', 'readSim']


def _createversionstring():
    from pkg_resources import get_distribution, DistributionNotFound
    # read version from installed metadata
    try:
        import os.path
        _dist = get_distribution('postpic')
        # Normalize case for Windows systems
        dist_loc = os.path.normcase(_dist.location)
        here = os.path.normcase(__file__)
        if not here.startswith(os.path.join(dist_loc, 'postpic')):
            # not installed, but there is another version that *is*
            raise DistributionNotFound
    except DistributionNotFound:
        __version__ = 'Please install this project with setup.py'
    else:
        __version__ = _dist.version

    # add Git description for __version__ if present
    try:
        import subprocess as sub
        import os.path
        cwd = os.path.dirname(__file__)
        p = sub.Popen(['git', 'describe', '--always', '--dirty'], stdout=sub.PIPE,
                      stderr=sub.PIPE, cwd=cwd)
        out, err = p.communicate()
        if not p.returncode:  # git exited without error
            __version__ += '_' + str(out)
    except OSError:
        # 'git' command not found
        pass
    return __version__

__version__ = _createversionstring()
