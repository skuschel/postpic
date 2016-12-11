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
'''
The plot subpackage should provide an interface to various plot backends.
'''
from __future__ import absolute_import, division, print_function, unicode_literals

plottercls = None


def use(plotcls):
    global plottercls
    if isinstance(plotcls, type('')):
        if plotcls in ['matplotlib', 'plotter_matplotlib']:
            from . import plotter_matplotlib
            plottercls = plotter_matplotlib.MatplotlibPlotter
        else:
            raise NameError('unknown type {:s}'.format(plotcls))
    else:
        plottercls = plotcls


# Default
use('matplotlib')
