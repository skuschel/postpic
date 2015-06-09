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
# Stephan Kuschel 2015

import abc
import collections
import numpy as np
from .. import helper
from .. import datahandling as dh
from ..analyzer import FieldAnalyzer

__all__ = ['Dumpreader_ifc', 'Simulationreader_ifc']


class Dumpreader_ifc(FieldAnalyzer):
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
        super(Dumpreader_ifc, self).__init__(self)
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
        ..helper.attribidentify
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
        name = {0: 'x', 1: 'y', 2: 'z'}[helper.axesidentify[axis]]
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



