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
from .. import helper
import os.path
import h5py
import glob

__all__ = ['SmileiReader', 'SmileiSeries']


def createh5indexfile(indexfile, fnames):
    '''
    Creates a h5 index file called indexfile containing external links to all h5
    datasets which are in in the h5 filelist fnames. This is fast as datasets
    will only be externally linked within the new indexfile.
    Therefore the indexfile will also be small in size.

    Will throw an error, if any of the h5 files contain a dataset under the same name.
    '''
    if os.path.isfile(indexfile):
        # indexfile already exists. do not recreate
        return

    dirname = osp.dirname(fnames[0])
    indexfile = osp.join(dirname, indexfile)

    def visitf(key):
        # key is a string
        # only link if key points to a dataset. Do not link groups
        if isinstance(hf[key], h5py._hl.dataset.Dataset):
            ih[key] = h5py.ExternalLink(fname, key)

    with h5py.File(indexfile, 'w') as ih:
        for fname in fnames:
            with h5py.File(fname, 'r') as hf:
                hf.visit(visitf)


class SmileiReader(OpenPMDreader):
    '''
    The Smilei reader will combine the information of different h5 files.
    Should work by pointing to the directory of the simulation
    `simreader = SmileiReader('path/to/simulation/*.h5', 1000)`,
    where 1000 points to iteration 1000 of the simulation.

    Args:
      h5file : String
        A String containing the relative Path to the h5 files of the simulation.
        Will be expanded using glob. Hidden files starting with `.` will be ignored.

    Kwargs:
      iteration : Integer
        An integer indicating the iteration to be loaded. Default is None, leading
        to the first iteration found in the h5file to be loaded.
    '''

    def __init__(self, h5filesstr, iteration=None):
        # The class given to super is the OpenPMDreader class, not the SmileiReader class.
        # This is on purpose to NOT run the `OpenPMDreader.__init__`.
        super(OpenPMDreader, self).__init__(h5filesstr)
        # Smilei uses multiple h5 files and also the iteration encoding differs from openPMD.
        h5files = glob.glob(h5filesstr)
        if len(h5files) == 0:
            raise IOError('No h5 files found matching {}'.format(h5filesstr))
        path = os.path.basename(h5filesstr)
        indexfile = os.path.join(path, '.postpic_smilei_index.h5')
        createh5indexfile(indexfile, h5files)

        self._h5 = h5py.File(indexfile, 'r')
        self._iteration = int(iteration)
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
        return '<SmileiReader at "{}" at iteration {:d}>'.format(self.dumpidentifier,
                                                                 self.timestep())


class SmileiSeries(Simulationreader_ifc):
    '''
    Reads a time series of dumps from a given directory.

    Point this to the directory and you will get all dumps,
    but possibly containing different data.
    `simreader = SmileiSeries('path/to/simulation/')`

    Or point this to a single file an you will only get the
    the iterations which are present in that file:
    `simreader = SmileiSeries('path/to/simulation/fields0.h5')`
    '''

    def __init__(self, h5file, dumpreadercls=SmileiReader, **kwargs):
        super(SmileiSeries, self).__init__(h5file, **kwargs)
        self.dumpreadercls = dumpreadercls
        self.h5file = h5file
        self.path = os.path.basename(h5file)
        if os.path.isfile(h5file):
            with h5py.File(h5file, 'r') as h5:
                self._dumpkeys = list(h5['data'].keys())
        elif os.path.isdir(h5file):
            pass  # TODO
        else:
            raise IOError('{} does not exist.'.format(h5file))

    def _getDumpreader(self, n):
        '''
        Do not use this method. It will be called by __getitem__.
        Use __getitem__ instead.
        '''
        return self.dumpreadercls(self.h5file, self._dumpkeys[n])

    def __len__(self):
        return len(self._dumpkeys)

    def __str__(self):
        return '<SmileiSeries based on "{}">'.format(self.simidentifier)
