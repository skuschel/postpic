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
# Stephan Kuschel, 2023
# Carolin Goll, 2023


from __future__ import absolute_import, division, print_function, unicode_literals

from . import OpenPMDreader
from . import Simulationreader_ifc
import numpy as np
import re
from .. import helper
from ..helper_fft import fft

__all__ = ['SmileiReader', 'SmileiSeries']


class SmileiReader(OpenPMDreader):
    '''
    The Reader implementation for Data written in the hdf5 file
    format following openPMD_ naming conventions.

    Args:
      h5file : String
        A String containing the relative Path to the .h5 file.

    Kwargs:
      iteration : Integer
        An integer indicating the iteration to be loaded. Default is None, leading
        to the first iteration found in the h5file to be loaded.
    '''

    def __init__(self, h5file, iteration=None):
        # The class given to super is the OpenPMDreader class, not the SmileiReader class.
        # This is on purpose to NOT run the `OpenPMDreader.__init__`.
        super(OpenPMDreader, self).__init__(h5file)
        # Smilei uses multiple h5 files and also the iteration encoding differs from openPMD.
        import os.path
        import h5py
        if not os.path.isfile(h5file):
            raise IOError('File "{}" doesnt exist.'.format(h5file))
        self._h5 = h5py.File(h5file, 'r')
        self._iteration = iteration
        if self._iteration is None:
            self._iteration = int(list(self._h5['data'].keys())[0])
        self._data = self._h5['/data/{:010d}/'.format(self._iteration)]
        self.attrs = self._data.attrs

# --- Level 0 methods ---

# --- Level 1 methods ---

# --- Level 2 methods ---

    def _keyE(self, component, **kwargs):
        # Smilei deviates from openPMD standard: Ex instead of E/x
        axsuffix = {0: 'x', 1: 'y', 2: 'z', 90: 'r', 91: 't'}[helper.axesidentify[component]]
        return 'fields/E{}'.format(axsuffix)

    def _keyB(self, component, **kwargs):
        # Smilei deviates from openPMD standard: Bx instead of B/x
        axsuffix = {0: 'x', 1: 'y', 2: 'z', 90: 'r', 91: 't'}[helper.axesidentify[component]]
        return 'fields/B{}'.format(axsuffix)

    def _simgridkeys(self):
        # Smilei deviates from openPMD standard: Ex instead of E/x
        return ['fields/Ex', 'fields/Ey', 'fields/Ez',
                'fields/Bx', 'fields/By', 'fields/Bz']

    def __str__(self):
        return '<SmileiReader at "{}", iteration {:d}>'.format(self.dumpidentifier,
                                                               self.timestep())


class SmileiSeries(Simulationreader_ifc):
    '''
    Reads a time series of dumps from a given directory.
    The simidentifier is expanded using glob in order to
    find matching files.
    '''

    def __init__(self, h5file, dumpreadercls=SmileiReader, **kwargs):
        super(SmileiSeries, self).__init__(h5file, **kwargs)
        self.dumpreadercls = dumpreadercls
        self.h5file = h5file
        if not os.path.isfile(h5file):
            raise IOError('File "{}" doesnt exist.'.format(h5file))
        with h5py.File(h5file, 'r') as h5:
            self._dumpkeys = list(h5['data'].keys())

    def _getDumpreader(self, n):
        '''
        Do not use this method. It will be called by __getitem__.
        Use __getitem__ instead.
        '''
        return self.dumpreadercls(self._dumpkeys[n])

    def __len__(self):
        return len(self._dumpkeys)

    def __str__(self):
        return '<SmileiSeries based on "{}">'.format(self.simidentifier)
