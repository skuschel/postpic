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

__version__ = '0.0.0'

# Use Git description for __version__ if present
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


