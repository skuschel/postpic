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
# Vinith Samson J, 2024

from __future__ import absolute_import, division, print_function, unicode_literals

from .openPMDh5 import OpenPMDreader
from . import Simulationreader_ifc
from .. import helper
import os.path
import h5py
import glob
import numpy as np
from .. import helper
from ..helper_fft import fft
import re

__all__ = ['SmileiReader', 'SmileiSeries']


def _generateh5indexfile(indexfile, fnames):
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

    def visitf(key):
        # key is a string
        if key.endswith('latest_IDs'):
            return
        # only link if key points to a dataset. Generally do not link groups.
        # However single scalars (identified by ´value in hf[key].attrs´)
        # maybe a group and must be linked as well.
        if isinstance(hf[key], h5py._hl.dataset.Dataset) or "value" in hf[key].attrs:
            ih[key] = h5py.ExternalLink(fname, key)
        elif isinstance(hf[key], h5py._hl.group.Group) and key not in ih:
            ih.create_group(key)
            for attr in hf[key].attrs:
                ih[key].attrs[attr] = hf[key].attrs[attr]

    with h5py.File(indexfile, 'w') as ih:
        for fname in fnames:
            with h5py.File(fname, 'r') as hf:
                hf.visit(visitf)


def _getindexfile(path):
    '''
    Returns the name of the index file after the file has been
    generated (File generation only if it doesnt exist)
    '''
    indexfile = os.path.join(path, '.postpic-smilei-index.h5')
    if not os.path.isfile(indexfile):
        # find all h5 files
        h5files = glob.glob(os.path.join(path, '*.h5'))
        print('generating index file "{}" from the following'
              'h5 files: {}'.format(indexfile, h5files))
        _generateh5indexfile(indexfile, h5files)
    return indexfile


class SmileiReader(OpenPMDreader):
    '''
    The Smilei reader can read  a single h5 file or combine the information of
    all h5 files in the same directory.
    Use either
    `simreader = SmileiReader('path/to/simulation/fields0.h5', 1000)`,
    where 1000 points to iteration 1000 of the simulation.
    or use
    `simreader = SmileiReader('path/to/simulation/', 1000)`
    to combine all h5 files in this directory.
    Internally the second command will create a new file called
    `.postpic-smilei-index.h5` which contains external links to all
    datasets in this directory.

    Args:
      h5file : String
        A String containing the relative Path to the h5 files of the simulation
        or the path to a single h5 file.
        Hidden files starting with `.` will be ignored.

      iteration : Integer
        An integer indicating the iteration to be loaded. Default is None, leading
        to the first iteration found in the h5file to be loaded.
    '''

    def __init__(self, h5file, iteration=None):
        # The class given to super is the OpenPMDreader class, not the SmileiReader class.
        # This is on purpose to NOT run the `OpenPMDreader.__init__`.
        super(OpenPMDreader, self).__init__(h5file)
        # Smilei uses multiple h5 files and also the iteration encoding differs from openPMD.
        if os.path.isfile(h5file):
            self._h5 = h5py.File(h5file, 'r')
        elif os.path.isdir(h5file):
            indexfile = _getindexfile(h5file)
            self._h5 = h5py.File(indexfile, 'r')
        else:
            raise IOError('"{}" is neither a h5 file nor a directory'
                          'containing h5 files'.format(h5file))

        if iteration is None:
            self._iteration = int(list(self._h5['data'].keys())[0])
        elif iteration not in [int(i) for i in list(self._h5['data'].keys())]:
            raise IOError("Iteration {} is in valid".format(iteration))
        else:
            self._iteration = int(iteration)

        self._data = self._h5['/data/{:010d}/'.format(self._iteration)]
        self.attrs = self._data.attrs

    @staticmethod
    def _modeexpansion_naiv(rawdata, theta=0):
        '''
        This method performes mode expansion of the raw data (an array consisting of complex
        numbers) for both single and multiple theta vaues.

        The output array has the shape (No.of theta, Nx, Nr)

        Args:
            rawdata : numpy array
            The elements of this array are complex numbers, this got structured from the
            raw data dumped in h5 file through getExpanded(key, theta) function.

            theta : float/integer OR list of floats/integer

        Output F_total is an array of real numbers which has shape (Np.of theta, Nx, Nr),
        this F_total is the real value summation of the fourier series.
        '''
        if np.array(theta).shape == ():
            theta = [theta]

        array_list = []

        (Nm, Nx, Nr) = rawdata.shape
        F_total = np.zeros((Nx, Nr))
        mode = [m for m in range(0, Nm)]
        for t in theta:

            for m in mode:
                F_total += np.real(rawdata[m])*np.cos(m*t)+np.imag(rawdata[m])*np.sin(m*t)
            array_list.append(F_total)

        mod_F_total = np.stack(array_list, axis=0)
        return mod_F_total

# --- Level 0 methods ---

    def _listAMmodes(self):
        '''
        This method is used to get the list of
        [prefix of field names, No.of AM modes] available in the dump.
        And it works only for AM mode technique.
        '''
        strings = np.array(self._data)
        mask = np.array(["_mode_" in s for s in strings])
        arr = strings[mask]
        if len(arr) == 0:
            return ([], [])
        max_suffix = float('-inf')
        max_suffix_string = None
        prefix_list = []

        for i in arr:
            prefix, suffix = i.split('_mode_')
            suffix_int = int(suffix)
            if suffix_int > max_suffix:
                max_suffix = suffix_int
                max_suffix_string = i
            prefix_list.append(prefix)

        availModes = [i for i in range(0, int(max_suffix_string[-1])+1)]

        # [field names prefix, available AM modes]
        return [prefix_list, availModes]

# --- Level 1 methods ---

    def _getExpanded(self, key, theta=0):
        '''
        _getExpanded() method converts the raw data real number array from h5file into
        a complex number array (This convertion is important while performing mode expansion)
        and finally returns the mode expanded array.

        This method takes input from the h5files dump which has the following format,
        field array = [[real_1,imag_1,real_2,image_2,.....],...]
        The shape of this field array is (Nx, 2x Nr)
        After real->complex conversion, the field array shape takes the form (Nx, Nr)
        This final array is fed into _modeexpansion_naiv method.
        '''
        array_list = []
        modes = self._listAMmodes()[-1]
        for mode in modes:

            field_name = key+"_mode_"+str(mode)
            field_array = np.array(self._data[field_name])
            field_array_shape = field_array.shape
            reshaped_array = field_array.reshape(field_array_shape[0], field_array_shape[1]//2, 2)
            complex_array = reshaped_array[:, :, 0] + 1j * reshaped_array[:, :, 1]
            array_list.append(complex_array)

        # Modified array of shape (Nmodes, Nx, Nr)
        mod_complex_data = np.stack(array_list, axis=0)
        factor = self._data["{}_mode_0".format(key)].attrs['unitSI']
        return SmileiReader._modeexpansion_naiv(rawdata=mod_complex_data, theta=theta)*factor

    def data(self, key, **kwargs):
        '''
        should work with any key, that contains data, thus on every hdf5.Dataset,
        but not on hdf5.Group. Will extract the data, convert it to SI and return it
        as a numpy array. Constant records will be detected and converted to
        a numpy array containing a single value only.

        If the key is in AM mode dump, then it performs the mode expansion.
        The theta values for which we need to perform
        mode expansion can be given as keyword args.

        Example:
            Data = data(key='El', theta=[0,pi/2,pi])
            Now the Data will array have the shape (3, Nx, Nr)
        '''

        # checking whether the key is in AM mode dump
        if key in self._listAMmodes()[0]:
            return self._getExpanded(key=key, **kwargs)
        else:
            record = self._data[key]

        if "value" in record.attrs:
            # constant data (a single int or float)
            ret = np.float64(record.attrs['value']) * record.attrs['unitSI']
        else:
            # array data
            ret = np.float64(record[()]) * record.attrs['unitSI']
        return ret

    # To get the offsets of the grid.
    def gridoffset(self, key, axis):
        axid = helper.axesidentify[axis]

        if axid == 91:  # theta
            return 0
        elif key in self._listAMmodes()[0] and axid in [0, 90]:
            key = "{}_mode_0".format(key)
            axid = int(axid/90)
            return super(SmileiReader, self).gridoffset(key=key, axis=axid)
        else:
            return super(SmileiReader, self).gridoffset(key, axis)

    # To get the grid spacing.
    def gridspacing(self, key, axis):
        axid = helper.axesidentify[axis]

        if key in self._listAMmodes()[0] and axid in [0, 90]:
            key = "{}_mode_0".format(key)
            axid = int(axid/90)
            return super(SmileiReader, self).gridspacing(key=key, axis=axid)
        else:
            return super(SmileiReader, self).gridspacing(key, axis)

    # To get the grid points
    def gridpoints(self, key, axis):
        axid = helper.axesidentify[axis]

        if key in self._listAMmodes()[0]:
            key = "{}_mode_0".format(key)
            (Nx, Nr) = self._data[key].shape
            Nr = Nr/2
            axid = int(axid/90)
            return (Nx, Nr)[axid]
        else:
            return super(SmileiReader, self).gridpoints(key=key, axis=axis)

# --- Level 2 methods ---

    def _keyE(self, component, **kwargs):
        # Smilei deviates from openPMD standard: Ex instead of E/x
        axsuffix = {0: 'x', 1: 'y', 2: 'z', 90: 'r', 91: 't'}[helper.axesidentify[component]]
        return 'E{}'.format(axsuffix)

    def _keyB(self, component, **kwargs):
        # Smilei deviates from openPMD standard: Bx instead of B/x
        axsuffix = {0: 'x', 1: 'y', 2: 'z', 90: 'r', 91: 't'}[helper.axesidentify[component]]
        return 'B{}'.format(axsuffix)

    def _simgridkeys(self):
        # Smilei deviates from openPMD standard: Ex instead of E/x
        return ['Ex', 'Ey', 'Ez', 'Er', 'El', 'Et',
                'Bx', 'By', 'Bz', 'Br', 'Bl', 'Bt',
                'Jx', 'Jy', 'Jz', 'Jr', 'Jl', 'Jt', 'Rho']

    def getSpecies(self, species, attrib):
        """
        Returns one of the attributes out of (x,y,z,px,py,pz,weight,ID,mass,charge) of
        this particle species.
        """
        attribid = helper.attribidentify[attrib]
        options = {9: 'particles/{}/weight',
                   0: 'particles/{}/position/x',
                   1: 'particles/{}/position/y',
                   2: 'particles/{}/position/z',
                   3: 'particles/{}/momentum/x',
                   4: 'particles/{}/momentum/y',
                   5: 'particles/{}/momentum/z',
                   10: 'particles/{}/id',
                   11: 'particles/{}/mass',
                   12: 'particles/{}/charge'}
        optionsoffset = {0: 'particles/{}/positionOffset/x',
                         1: 'particles/{}/positionOffset/y',
                         2: 'particles/{}/positionOffset/z'}
        key = options[attribid]
        offsetkey = optionsoffset.get(attribid)
        try:
            data = self.data(key.format(species))
            if offsetkey is not None:
                data += self.data(offsetkey.format(species))
            ret = np.asarray(data, dtype=np.float64)
        except IndexError:
            raise KeyError
        return ret

    def getderived(self):
        '''
        return all other fields dumped, except E and B.
        '''
        ret = []
        self['.'].visit(ret.append)
        # TODO: remove E and B fields and particles from list
        ret.sort()
        return ret

    def __str__(self):
        return '<SmileiReader at "{}" at iteration {:d}>'.format(self.dumpidentifier,
                                                                 self.timestep())


class SmileiSeries(Simulationreader_ifc):
    '''
    Reads a time series of dumps from a given directory.

    Point this to the directory and you will get all dumps,
    but possibly containing different data.
    `simreader = SmileiSeries('path/to/simulation/')`

    Alternatively point this to a single file and you will only get
    the iterations which are present in that file:
    `simreader = SmileiSeries('path/to/simulation/Fields0.h5')`
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
            indexfile = _getindexfile(h5file)
            self._h5 = h5py.File(indexfile, 'r')
            with h5py.File(indexfile, 'r') as h5:
                self._dumpkeys = [int(i) for i in h5['data'].keys()]
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
