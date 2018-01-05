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

  *The open source particle-in-cell post processor.*

Particle-in-cell simulations are a valuable tool for the simulation of non-equelibrium
systems in plasma- or astrophysics.
Such simulations usually produce a large amount of data consisting of electric and magnetic
field data as well as particle positions and momenta. While there are various PIC codes freely
available, the task of post-processing -- essentially condensing the large amounts of data
into small units suitable for plotting routines -- is typically left to each user individually.
As post-processing may be a time consuming and error-prone process,
this python package has been developed.

*Postpic* can handle two different types of data:

Field data
  which is data sampled on a predefined grid, such as electic and magnetic fields, particle- or
  charge densities, currents, etc.
  Fields are usually the data, which can be plotted directly.
  See :class:`postpic.Field`.

Particle data
  which is data of multiple particles and for each particle positions (`x`, `y`, `z`) and
  momenta (`px`, `py`, `pz`) are known. Particles usually also have `weight`,
  `charge`, `time` and a unique `id`.
  Postpic can transform particle data to field data using the same algorithm and particle shapes,
  which are used in most PIC Simulations. The particle-to-grid routines are written in C
  for maximum performance. See :class:`postpic.MultiSpecies`.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

from . import helper
from . import datahandling
from . import particles
from . import experimental
from . import plotting
from . import io
from .datahandling import *
from .helper import *
from .particles import *
from . import datareader
from .datareader import chooseCode, readDump, readSim
from ._version import get_versions
from .io import *

__all__ = ['helper']
__all__ += datahandling.__all__
__all__ += helper.__all__
__all__ += particles.__all__
__all__ += ['datareader', 'plotting']
# high level functions
__all__ += ['chooseCode', 'readDump', 'readSim']
__all__ += io.__all__

__version__ = get_versions()['version']
__git_version__ = get_versions()['full-revisionid']
del get_versions
