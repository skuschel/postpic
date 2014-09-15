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

import datareader
import analyzer
import plotting

__all__ = ['datareader', 'analyzer', 'plotting']

# read version from installed metadata
from pkg_resources import get_distribution, DistributionNotFound
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
    p = sub.Popen(['git', 'describe', '--always'], stdout=sub.PIPE,
                  stderr=sub.PIPE, cwd=cwd)
    out, err = p.communicate()
    if not p.returncode:  # git exited without error
        __version__ += '_g' + out
except OSError:
    # 'git' command not found
    pass


