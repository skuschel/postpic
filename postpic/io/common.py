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
# Stefan Tietze, 2017
# Alexander Blinne, 2017
# Stephan Kuschel, 2017
'''
The postpic.io module provides free functions for importing and exporting data.
'''


def _header_string():
    '''
    creates a header string for general export information.
    '''
    from .. import __version__
    import datetime
    now = str(datetime.datetime.now())
    ret = '''
    This file was written by postpic {v:}
    --- the open-source particle-in-cell post-processor. ---
    https://github.com/skuschel/postpic

    written on {now:}\n
    '''
    return ret.format(v=__version__, now=now)
