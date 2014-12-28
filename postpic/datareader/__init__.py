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
# Stephan Kuschel 2014
'''
The Datareader package contains methods and interfaces to read data from
any Simulation.

The basic concept consits of two different types of readers:

The Dumpreader
--------------
This has to be subclassed from Dumpreader_ifc and allows to read a single dump
created by the simulation. To identify which dump should be read its
initialized with a dumpidentifier. This dumpidentifier can be almost anything,
but in the easiest case this is the filepath pointing to a single file
containing every information about this simulation dump. With this information
the dumpreader must be able to read all data regarding this dump (which is a
lot: X, Y, Z, Px, Py, Py, weight, mass, charge, ID,.. for all particle species,
electric and magnetic fields on grid, the grid itself, mabe particle ids,...)

The Simulationreader
--------------------
This has to be subclassed from Simulationreader_ifc and allows to read a full
list of simulation dumps. Thus an alternate Name for this class could be
"Dumpsequence". This allows the code to track particles from different times
of the simulation or create plots with a time axis.

Stephan Kuschel 2014
'''


import abc
import collections
import numpy as np
from .. import _const
from .. import datahandling as dh

# --- Interface ---


class Dumpreader_ifc(object):
    '''
    Interface class for reading a single dump. A dump contains informations
    about the  simulation at a single timestep (Usually E- and B-Fields on
    grid + particles).

    Any Dumpreader_ifc implementation will always be initialized using a
    dumpidentifier. This dumpidentifier can be anything, that points to
    the data of a dump. In the easiest case this is just the filename of
    the dump holding all data of that timestep
    (for example .sdf file for EPOCH, .hdf5 for some other code).

    The dumpreader should provide all necessary informations in a unified interface
    but at the same time it should not restrict the user to these properties of there
    dump only. The recommended implementation is shown here (EPOCH and VSim reader
    work like this):
    All (!) data, that is saved in a single dump should be accessible via the
    self.__getitem__(key) method. Together with the self.keys() method, this will ensure,
    that every dumpreader works as a dictionary and every dump attribute is accessible
    via this dictionary.

    All other functions, which provide a unified interface should just point to the
    right key. If some attribute wasnt dumped a KeyError must be thrown. This allows
    classes which are using the reader to just exit if a needed property wasnt dumped
    or to catch the KeyError and proceed by actively ignoring it.

    It is highly recommended to also override the __str__ function.

    Args:
      dumpidentifier : variable type
        whatever identifies the dump. It is recommended to use a String
        here pointing to a file.
    '''
    __metaclass__ = abc.ABCMeta

    def __init__(self, dumpidentifier, name=None):
        self.dumpidentifier = dumpidentifier
        self._name = name

    @abc.abstractmethod
    def keys(self):
        '''
        Returns:
          a list of keys, that can be used in __getitem__ to read
          any information from this dump.
        '''
        pass

    @abc.abstractmethod
    def __getitem__(self, key):
        pass

    # --- General Information ---
    @abc.abstractmethod
    def timestep(self):
        pass

    @abc.abstractmethod
    def time(self):
        pass

    @abc.abstractmethod
    def simdimensions(self):
        pass

    # --- Data on Grid ---
    @abc.abstractmethod
    def dataE(self, axis):
        pass

    @abc.abstractmethod
    def dataB(self, axis):
        pass

    @abc.abstractmethod
    def grid(self, axis):
        pass

    # --- Particle Data ---
    @abc.abstractmethod
    def listSpecies(self):
        pass

    @abc.abstractmethod
    def getSpecies(self, species, attrib):
        '''
        This function gives access to any of the particle properties in
        .._const.attribidentify
        This method can behave in the following ways:
        1) Return a list of scalar properties for each particle of this species
        2) Return a single float (i.e. `1.2`, NOT `[1.2]`) to show that
           every particle of this species has the same scalar value to thisdimmax
           property assigned. This might be quite often used for charge or mass
           that are defined per species.
        3) Raise a KeyError if the requested property or species wasn dumped.
        '''
        pass

    @property
    def name(self):
        if self._name:
            ret = self._name
        else:
            ret = str(self.dumpidentifier)
        return ret

    @name.setter
    def name(self, val):
        self._name = str(val) if bool(val) else None

    def __str__(self):
        return '<Dumpreader initialized with "' \
               + str(self.dumpidentifier) + '">'

    # Higher Level Functions for usability

    def getaxis(self, axis):
        '''
        Args:
          axis : string or int
            the axisidentifier

        Returns: an Axis object for a given axis.
        '''
        name = {0: 'x', 1: 'y', 2: 'z'}[_const.axesidentify[axis]]
        ret = dh.Axis(name=name, unit='m')
        ret.grid = self.grid(axis)
        return ret

    def extent(self, axis):
        '''
        Args:
          axis : string or int
            the axisidentifier

        Returns: the extent of the simulation for a given axis.
        '''
        ax = self.getaxis(axis)
        return np.array(ax.extent)

    def gridpoints(self, axis):
        '''
        Args:
          axis : string or int
            the axisidentifier

        Returns: the number of grid points along a given axis.
        '''
        return len(self.grid(axis))

    def getspacialresolution(self, axis):
        '''
        Args:
          axis : string or int
            the axisidentifier

        Returns: the spacial grid resolution along a given axis.
        '''
        extent = self.extent(axis)
        return (extent[1] - extent[0]) / float(self.gridpoints(axis))


class Simulationreader_ifc(collections.Sequence):
    '''
    Interface for reading the data of a full Simulation.

    Any Simulationreader_ifc implementation will always be initialized using a
    simidentifier. This simidentifier can be anything, that points to
    the data of multiple dumps. In the easiest case this can be the .visit
    file.

    The Simulationreader_ifc is subclass of collections.Sequence and will
    thus behave as a Sequence. The Objects in the Sequence are supposed to be
    subclassed from Dumpreader_ifc.

    It is highly recommended to also override the __str__ function.

    Args:
      simidentifier : variable type
        something identifiying a series of dumps.
    '''

    def __init__(self, simidentifier, name=None):
        self.simidentifier = simidentifier
        self._name = name

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[ii] for ii in xrange(*key.indices(len(self)))]
        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            return self.getDumpreader(key)
        else:
            raise TypeError("Invalid argument type.")

    @abc.abstractmethod
    def getDumpreader(self, number):
        '''
        :returns: the corresponding Dumpreader.
        '''
        pass

    @property
    def name(self):
        if self._name:
            ret = self._name
        else:
            ret = str(self.simidentifier)
        return ret

    @name.setter
    def name(self, val):
        self._name = str(val) if bool(val) else None

    @abc.abstractmethod
    def __len__(self):
        pass

    def __str__(self):
        return '<Simulationreader initialized with "' \
               + str(self.simidentifier) + '">'

    # Higher Level Functions for usability

    def times(self):
        return np.array([s.time() for s in self])

# --- datareader package functions ---

__all__ = ['Dumpreader_ifc', 'Simulationreader_ifc', 'chooseCode',
           'readDump', 'readSim']

_dumpreadercls = None
_simreadercls = None


def setdumpreadercls(dumpreadercls):
    '''
    Sets the class that is used for reading dumps.
    dumpreadercls needs to be subclass of "Dumpreader_ifc".
    '''
    if issubclass(dumpreadercls, Dumpreader_ifc):
        global _dumpreadercls
        _dumpreadercls = dumpreadercls
    else:
        raise Exception('In order to set a reader class for a new file'
                        ' format it needs to be subclass of "Dumpreader_ifc"')


def setsimreadercls(simreadercls):
    '''
    Sets the class that is used for reading dumps.
    simreadercls needs to be subclass of "Simulationreader_ifc".
    '''
    if issubclass(simreadercls, Simulationreader_ifc):
        global _simreadercls
        _simreadercls = simreadercls
    else:
        raise Exception('In order to set a reader for a new file'
                        ' format it needs to be subclass of '
                        '"Simulationreader_ifc"')


def readDump(dumpidentifier, **kwargs):
    global _dumpreadercls
    if _dumpreadercls is None:
        raise Exception('Specify dumpreaderclass first.')
    return _dumpreadercls(dumpidentifier, **kwargs)


def readSim(simidentifier, **kwargs):
    global _simreadercls
    if _simreadercls is None:
        raise Exception('Specify simreaderclass first.')
    return _simreadercls(simidentifier, **kwargs)


def chooseCode(code):
    '''
    Chooses appropriate reader for the given simulation code.

    Args:
      code : string
        Possible options are:
          - "DUMMY": dummy class creating fake data.
          - "EPOCH": .sdf files written by EPOCH1D, EPOCH2D or EPOCH3D.
          - "VSIM": .hdf5 files written by VSim.
    '''
    if code in ['EPOCH', 'epoch', 'EPOCH1D', 'EPOCH2D', 'EPOCH3D']:
        from epochsdf import Sdfreader, Visitreader
        setdumpreadercls(Sdfreader)
        setsimreadercls(Visitreader)
    elif code in ['VSim', 'VSIM', 'vsim']:
        from vsimhdf5 import Hdf5reader, VSimReader
        setdumpreadercls(Hdf5reader)
        setsimreadercls(VSimReader)
    elif code in ['DUMMY', 'dummy']:
        from dummy import Dummyreader, Dummysim
        setdumpreadercls(Dummyreader)
        setsimreadercls(Dummysim)
    else:
        raise TypeError('Code "' + str(code) + '" not recognized.')














