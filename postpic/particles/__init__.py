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

from __future__ import absolute_import, division, print_function, unicode_literals
import sys

from . import particles
from .particles import *
from . import scalarproperties
from .scalarproperties import ScalarProperty
from . import _routines
from ._routines import *
from ._routines import particleshapes


__all__ = ['ScalarProperty']
__all__ += particles.__all__
__all__ += _routines.__all__


if sys.version_info[0] == 2:
    # some weired python2 thing
    __all__ = [n.encode('ascii') for n in __all__]
