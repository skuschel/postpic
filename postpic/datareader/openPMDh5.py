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
# Copyright Stephan Kuschel, 2018-2019
'''
.. _openPMD: https://github.com/openPMD/openPMD-standard

Support for hdf5 files following the openPMD_ Standard.

Dependecies:
  - h5py: read hdf5 files with python

Written by Stephan Kuschel 2016
'''
from __future__ import absolute_import, division, print_function, unicode_literals

from . import Dumpreader_ifc
from . import Simulationreader_ifc
import numpy as np
import re
from .. import helper

__all__ = ['OpenPMDreader', 'FbpicReader', 'FileSeries']


class OpenPMDreader(Dumpreader_ifc):
    '''
    The Reader implementation for Data written in the hdf5 file
    format following openPMD_ naming conventions.

    Args:
      h5file : String
        A String containing the relative Path to the .h5 file.
    '''

    def __init__(self, h5file, **kwargs):
        super().__init__(h5file, **kwargs)
        import os.path
        import h5py
        if not os.path.isfile(h5file):
            raise IOError('File "' + str(h5file) + '" doesnt exist.')
        self._h5 = h5py.File(h5file, 'r')
        self._iteration = int(list(self._h5['data'].keys())[0])
        self._data = self._h5['/data/{:d}/'.format(self._iteration)]
        self.attrs = self._data.attrs

    def __del__(self):
        del self._data

# --- Level 0 methods ---

    def keys(self):
        return list(self._data.keys())

    def __getitem__(self, key):
        return self._data[key]

# --- Level 1 methods ---

    def data(self, key):
        '''
        should work with any key, that contains data, thus on every hdf5.Dataset,
        but not on hdf5.Group. Will extract the data, convert it to SI and return it
        as a numpy array. Constant records will be detected and converted to
        a numpy array containing a single value only.
        '''
        record = self[key]
        if "value" in record.attrs:
            # constant data (a single int or float)
            ret = np.float64(record.attrs['value']) * record.attrs['unitSI']
        else:
            # array data
            ret = np.float64(record[()]) * record.attrs['unitSI']
        return ret

    def gridoffset(self, key, axis):
        axid = helper.axesidentify[axis]
        if "gridUnitSI" in self[key].attrs:
            attrs = self[key].attrs
        else:
            attrs = self[key].parent.attrs
        return attrs['gridGlobalOffset'][axid] * attrs['gridUnitSI']

    def gridspacing(self, key, axis):
        axid = helper.axesidentify[axis]
        if "gridUnitSI" in self[key].attrs:
            attrs = self[key].attrs
        else:
            attrs = self[key].parent.attrs
        return attrs['gridSpacing'][axid] * attrs['gridUnitSI']

    def gridpoints(self, key, axis):
        axid = helper.axesidentify[axis]
        return self[key].shape[axid]

# --- Level 2 methods ---

    def timestep(self):
        return self._iteration

    def time(self):
        return np.float64(self.attrs['time'] * self.attrs['timeUnitSI'])

    def simdimensions(self):
        '''
        the number of spatial dimensions the simulation was using.
        '''
        for k in self._simgridkeys():
            try:
                gs = self.gridspacing(k, None)
                return len(gs)
            except(KeyError):
                pass
        raise KeyError('number of simdimensions could not be retrieved for {}'.format(self))

    def _keyE(self, component, **kwargs):
        axsuffix = {0: 'x', 1: 'y', 2: 'z', 90: 'r', 91: 't'}[helper.axesidentify[component]]
        return 'fields/E/' + axsuffix

    def _keyB(self, component, **kwargs):
        axsuffix = {0: 'x', 1: 'y', 2: 'z', 90: 'r', 91: 't'}[helper.axesidentify[component]]
        return 'fields/B/' + axsuffix

    def _simgridkeys(self):
        return ['fields/E/x', 'fields/E/y', 'fields/E/z',
                'fields/B/x', 'fields/B/y', 'fields/B/z']

    def listSpecies(self):
        ret = list(self['particles'].keys())
        return ret

    def getSpecies(self, species, attrib):
        """
        Returns one of the attributes out of (x,y,z,px,py,pz,weight,ID,mass,charge) of
        this particle species.
        """
        attribid = helper.attribidentify[attrib]
        options = {9: lambda s: self.data('particles/' + s + '/weighting'),
                   0: lambda s: self.data('particles/' + s + '/position/x') +
                   self.data('particles/' + s + '/positionOffset/x'),
                   1: lambda s: self.data('particles/' + s + '/position/y') +
                   self.data('particles/' + s + '/positionOffset/y'),
                   2: lambda s: self.data('particles/' + s + '/position/z') +
                   self.data('particles/' + s + '/positionOffset/z'),
                   3: lambda s: self.data('particles/' + s + '/momentum/x'),
                   4: lambda s: self.data('particles/' + s + '/momentum/y'),
                   5: lambda s: self.data('particles/' + s + '/momentum/z'),
                   10: lambda s: self.data('particles/' + s + '/id'),
                   11: lambda s: self.data('particles/' + s + '/mass'),
                   12: lambda s: self.data('particles/' + s + '/charge')}
        try:
            ret = np.float64(options[attribid](species))
        except(IndexError):
            raise KeyError
        return ret

    def getderived(self):
        '''
        return all other fields dumped, except E and B.
        '''
        ret = []
        self['fields'].visit(ret.append)
        ret = ['fields/' + r for r in ret if not (r.startswith('E') or r.startswith('B'))]
        ret = [r for r in ret if hasattr(self[r], 'value')]
        ret.sort()
        return ret

    def __str__(self):
        return '<OpenPMDh5reader at "' + str(self.dumpidentifier) + '">'


class FbpicReader(OpenPMDreader):
    '''
    Special OpenPMDreader for FBpic, which is using an expansion into radial modes.

    This is subclass of the OpenPMDreader which is converting the modes to
    a radial representation.
    '''
    def __init__(self, simidentifier, **kwargs):
        super().__init__(simidentifier, **kwargs)

    @staticmethod
    def modeexpansion(rawdata, theta=0):
        '''
        The mode representation will be expanded for a given theta.
        rawdata has to have the shape (Nm, Nr, Nz).
        the returned array will be of shape (Nr, Nz).
        '''
        rawdata = np.asarray(rawdata)
        (Nm, Nr, Nz) = rawdata.shape
        mult_above_axis = [1]
        for mode in range(1, int(Nm / 2) + 1):
            cos = np.cos(mode * theta)
            sin = np.sin(mode * theta)
            mult_above_axis += [cos, sin]
        mult_above_axis = np.asarray(mult_above_axis)
        F_total = np.tensordot(mult_above_axis,
                               rawdata, axes=(0, 0))
        assert F_total.shape == (Nr, Nz), \
            '''
            Assertion error. Please open a new issue on github to report this.
            shape={}, Nr={}, Nz={}
            '''.format(F_total.shape, Nr, Nz)
        return F_total

    @staticmethod
    def _radialdata(component, rawdata, theta=0):
        '''
        converts to radial data using `modeexpansion`, possibly for multiple
        theta at once.
        '''
        if np.asarray(theta).shape is ():
            # single theta
            data = FbpicReader.modeexpansion(rawdata, theta=theta)
        else:
            # multiple theta
            data = np.asarray([FbpicReader.modeexpansion(rawdata, theta=t) for t in theta])
        return data

    # override inherited method to count points after mode expansion
    def gridoffset(self, key, axis):
        axid = helper.axesidentify[axis]
        if axid == 91:  # theta
            return 0
        else:
            # r, theta, z
            axidremap = {90: 0, 2: 1}[axid]
            return super().gridoffset(key, axidremap)

    # override inherited method to count points after mode expansion
    def gridspacing(self, key, axis):
        axid = helper.axesidentify[axis]
        if axid == 91:  # theta
            return 2 * np.pi / self.gridpoints(key, axis)
        else:
            # r, theta, z
            axidremap = {90: 0, 2: 1}[axid]
            return super().gridspacing(key, axidremap)

    # override inherited method to count points after mode expansion
    def gridpoints(self, key, axis):
        axid = helper.axesidentify[axis]
        axid = axid % 90  # for r and theta
        (Nm, Nr, Nz) = self[key].shape
        # Ntheta does technically not exists because of the mode
        # representation. To do a proper conversion from the modes to
        # the grid, choose Ntheta based on the number of modes.
        # Nyquist should be `Ntheta = 2*Nm + 1`
        Ntheta = 2 * Nm + 1
        return (Nr, Ntheta, Nz)[axid]

    def rawdataE(self, component, **kwargs):
        return np.float64(self.data(self._keyE(component, **kwargs)))

    @helper.append_doc_of(modeexpansion)
    def dataE(self, component, theta=0, **kwargs):
        '''
        Return the electric field data.

        Circular modes will be expanded for the given `theta`, using the
        modeexpansion for a single `theta` as shown below. If there are multiple
        `theta` given, the result will be stacked in the first dimension.
        '''
        raw = self.rawdataE(component, **kwargs)
        data = self._radialdata('E{}'.format(component), raw, theta=theta)
        return data

    def rawdataB(self, component, **kwargs):
        return np.float64(self.data(self._keyB(component, **kwargs)))

    @helper.append_doc_of(modeexpansion)
    def dataB(self, component, theta=0, **kwargs):
        '''
        Return the magnetic field data.

        Circular modes will be expanded for the given `theta`, using the
        modeexpansion for a single `theta` as shown below. If there are multiple
        `theta` given, the result will be stacked in the first dimension.
        '''
        raw = self.rawdataB(component, **kwargs)
        data = self._radialdata('B{}'.format(component), raw, theta=theta)
        return data

    # override
    def __str__(self):
        return '<FbpicReader at "' + str(self.dumpidentifier) + '">'


class FileSeries(Simulationreader_ifc):
    '''
    Reads a time series of dumps from a given directory.
    The simidentifier is expanded using glob in order to
    find matching files.
    '''

    def __init__(self, simidentifier, dumpreadercls=OpenPMDreader, **kwargs):
        super().__init__(simidentifier, **kwargs)
        self.dumpreadercls = dumpreadercls
        import glob
        self._dumpfiles = glob.glob(simidentifier)
        self._dumpfiles.sort()

    def _getDumpreader(self, n):
        '''
        Do not use this method. It will be called by __getitem__.
        Use __getitem__ instead.
        '''
        return self.dumpreadercls(self._dumpfiles[n])

    def __len__(self):
        return len(self._dumpfiles)

    def __str__(self):
        return '<FileSeries at "' + self.simidentifier + '">'
