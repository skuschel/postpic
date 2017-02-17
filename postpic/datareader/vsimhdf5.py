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
# Georg Wittig, Stephan Kuschel 2014

'''
Reader for HDF5 File format written by the VSim Code:
http://www.txcorp.com/support/vsim-support-menu/vsim-documentation
Dependecies:
h5py  The Python actual reader for hdf5 file format.
Georg Wittig, Stephan Kuschel 2014
'''
from __future__ import absolute_import, division, print_function, unicode_literals

from . import Dumpreader_ifc
from . import Simulationreader_ifc
from .. import helper


import h5py
import numpy as np
import os

__all__ = ['Hdf5reader', 'VSimReader']


class Hdf5reader(Dumpreader_ifc):
    '''
    The Reader implementation for HDF5 Data written by the VSim Code.
    as argument h5file can be any *.h5 file of the dump of consideration.
    '''

    def __init__(self, h5file, **kwargs):
        '''
        Initializes the Hdf5reader for a specific h5file.
        '''
        super(self.__class__, self).__init__(h5file, **kwargs)
        if not os.path.isfile(h5file):
            raise IOError('File "' + str(h5file) + '" doesnt exist.')
        pathname = os.path.abspath(os.path.dirname(h5file))
        filelist = [os.path.join(pathname, f) for f in os.listdir(pathname) if f.endswith(".h5")]
        self._time = h5py.File(h5file)["time"].attrs["vsTime"]
        # all dumped h5 files at the same time
        self._dumplist = [f for f in filelist
                          if self._time == h5py.File(f)["time"].attrs["vsTime"]]

    def keys(self):
        keys = []
        for f in self._dumplist:
            for a in h5py.File(f):
                if a not in keys:
                    keys.append(a)
        return keys

    def __getitem__(self, key):
        ''' delivers one dataset with the key key.'''
        for f in self._dumplist:
            for a in h5py.File(f):
                if a == key:
                    return h5py.File(f)[key]
        # print " couldn't find key ", key
        return None

    def timestep(self):
        return self["time"].attrs['vsStep']

    def time(self):
        return self._time

    def simdimensions(self):
        return self["compGridGlobal"].attrs["vsNumCells"].shape[0]

    def dataE(self, axis, **kwargs):
        # x, y, z, px, py, pz same as in sdf. weigt and ID not included.
        axis = helper.axesidentify[axis]
        try:
            return np.float64(self["ElecMultiField"][..., axis])
        except(KeyError):
            return None

    def dataB(self, axis, **kwargs):
        # x, y, z, px, py, pz same as in sdf. wweigt and ID not included.
        axis = helper.axesidentify[axis]
        try:
            return np.float64(self["MagMultiField"][..., axis])
        except(KeyError):
            return None

    def grid(self, axis):
        ''' returns the array of the positions of all cells on axis = axis.  '''
        # x, y, z, px, py, pz same as in sdf. weigt and ID not included.
        axis = helper.axesidentify[axis]
        temp = self["compGridGlobal"]
        return np.linspace(temp.attrs["vsLowerBounds"][axis],
                           temp.attrs["vsUpperBounds"][axis], temp.attrs["vsNumCells"][axis])

    def listSpecies(self):
        ''' returns all h5 dumps that have a attribute "mass" '''
        specieslist = []
        for f in self._dumplist:
            h5 = h5py.File(f)
            for key in h5:
                if 'mass' in h5[key].attrs:
                    specieslist.append(key)
        return specieslist

    def getSpecies(self, species, attrib):
        '''
        Returns one of the attributes out of (x,y,z,px,py,pz,weight,ID) of
        this particle species.
        Valid Scalar attributes are (mass, charge).
        '''
        # x, y, z, px, py, pz same as in sdf. weigt and ID not included.
        attrib = helper.attribidentify[attrib]
        attrib = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 9: 'numPtclsInMacro',
                  11: 'mass', 12: 'charge'}[attrib]
        if isinstance(attrib, int):
            ret = np.float64(self[species])[:, attrib]
            # VSim dumps gamma*v = p/m0, so multiply by mass if px, pz or pz requested
            if attrib > 2:
                ret = ret * self.getSpecies(species, 'mass')
            return ret
        else:
            return np.float64(self[species].attrs[attrib])

    def getderived(self):
        '''
        Returns all Keys starting with "Derived/".
        '''
        pass

    def __str__(self):
        return '<Hdf5reader at "' + str(self.dumpidentifier) + '">'


class VSimReader(Simulationreader_ifc):
    '''
    Represents a full Simulation ( = Series of Dumps with equal output). The VSimReader must be
    initialized with the path to a folder containing all the dumps. It will then walk through
    all available .h5 files in this directory to identify the available timesteps of the
    simulation.
    '''

    def __init__(self, path, **kwargs):
        super(self.__class__, self).__init__(path, **kwargs)
        self.path = path
        import os.path
        if not os.path.isdir(path):
            raise IOError('Path "' + str(path) + '" is no directory.')
        # check every .h5 file in directory path
        self._filelist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".h5")]
        # only use dumps with time > 0
        h5s = [h5py.File(f) for f in self._filelist]
        boollist = [h5['time'].attrs['vsTime'] > 0 for h5 in h5s]
        self._filelist = [self._filelist[i] for i in range(len(self._filelist)) if boollist[i]]
        h5s = [h5s[i] for i in range(len(h5s)) if boollist[i]]
        # find all possible timesteps
        self._steplist = [h5['time'].attrs['vsStep'] for h5 in h5s]
        self._stepsdumped = list(set(self._steplist))
        self._stepsdumped.sort()

    def __len__(self):
        return len(self._stepsdumped)

    def getDumpreader(self, index):
        fileindex = self._steplist.index(self._stepsdumped[index])
        return Hdf5reader(self._filelist[fileindex])

    def __str__(self):
        return '<VSimReader at "' + str(self.path) + '">'
