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
# Alexander Blinne, 2017

from __future__ import absolute_import, division, print_function, unicode_literals
from future.utils import with_metaclass

import abc
import collections
import warnings
import numpy as np
from .. import helper
from .._field_calc import FieldAnalyzer

__all__ = ['Dumpreader_ifc', 'Simulationreader_ifc']


class Dumpreader_ifc(with_metaclass(abc.ABCMeta, FieldAnalyzer)):
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

    Hirachy of methods
    ------------------
    * Level 0:
    __getitem__ and keys(self) are level 0 methods, meaning it must be possible to access
    everthing with those methods.

    * Level 1:
    provide direct data access by forwarding the requests to the corresponding Level 0
    or Level 1 methods.

    * Level 2:
    provide user access to the data by forwarding the request to Level 1 or Level 2 methods,
    but NOT to Level 0 methods.

    If some attribute wasnt dumped a KeyError must be thrown. This allows
    classes which are using the reader to just exit if a needed property wasnt dumped
    or to catch the KeyError and proceed by actively ignoring it.

    It is highly recommended to also override the functions __str__ and gridpoints.

    Args:
      dumpidentifier : variable type
        whatever identifies the dump. It is recommended to use a String
        here pointing to a file.
    '''

    def __init__(self, dumpidentifier, name=None):
        super(Dumpreader_ifc, self).__init__()
        self.dumpidentifier = dumpidentifier
        self._name = name

# --- Level 0 methods ---

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
        '''
        access to everything. May return hd5 objects corresponding to the "key".
        '''
        pass

# --- Level 1 methods ---

    @abc.abstractmethod
    def data(self, key):
        '''
        access to every raw data. needs to return numpy arrays corresponding to the "key".
        '''
        pass

    @abc.abstractmethod
    def gridoffset(self, key, axis):
        '''
        offset of the beginning of the first cell of the grid.
        '''
        pass

    @abc.abstractmethod
    def gridspacing(self, key, axis):
        '''
        size of one grid cell in the direction "axis".
        '''
        pass

    def gridpoints(self, key, axis):
        '''
        Number of grid points along "axis". It is highly recommended to override this
        method due to performance reasons.
        '''
        warnings.warn('Method "gridpoints(self, key, axis)" is not overridden in datareader. '
                      'This is may alter performance.')
        return self.data(key).shape[helper.axesidentify[axis]]

    def gridnode(self, key, axis):
        '''
        The grid nodes along "axis". Grid nodes include the beginning and the end of the grid.
        Example: If the grid has 20 grid points, it has 21 grid nodes or grid edges.
        '''
        offset = self.gridoffset(key, axis)
        n = self.gridpoints(key, axis)
        return np.linspace(offset,
                           offset + self.gridspacing(key, axis) * n,
                           n + 1)

# --- Level 2 methods ---

    # --- General Information ---
    @abc.abstractmethod
    def timestep(self):
        pass

    @abc.abstractmethod
    def time(self):
        pass

    @abc.abstractmethod
    def simdimensions(self):
        '''
        the number of spatial dimensions the simulations was using.
        Must be 1, 2 or 3.
        '''
        pass

    # --- Data on Grid ---
    # _key[E,B] methods are ONLY used inside the datareader class
    @abc.abstractmethod
    def _keyE(self, component, **kwargs):
        '''
        The key where the E field component can be found.
        kwargs will be forwarded and can be used here to specify that alternate
        keys will be used instead. For example you might have dumped a default Ex
        field (no kwargs), but also another one with low resolution (lowres=2)
        and another one with low resolution and averaged over some laser periods
        (lowres=3, average=100).
        The naming of those kwargs is reader specific.
        '''
        pass

    @abc.abstractmethod
    def _keyB(self, component, **kwargs):
        '''
        The key where the B field component can be found.
        see _keyE for a description of kwargs.
        '''
        pass

    # if you need to customize more, just skip _key[E,B] methods and
    # override the following 4 methods to have full control.
    def dataE(self, component, **kwargs):
        return np.float64(self.data(self._keyE(component, **kwargs)))

    def gridkeyE(self, component, **kwargs):
        return self._keyE(component, **kwargs)

    def dataB(self, component, **kwargs):
        return np.float64(self.data(self._keyB(component, **kwargs)))

    def gridkeyB(self, component, **kwargs):
        return self._keyB(component, **kwargs)

    def _simgridkeys(self):
        '''
        returns a list of keys that can be tried one after another to determine the grid,
        that the actual simulations was running on.
        This is dirty. Rather override self.simgridpoints and self.simextent
        with your own (better performance) implementation.
        '''
        return []

    def simgridpoints(self, axis):
        for key in self._simgridkeys():
            try:
                return self.gridpoints(key, axis)
            except(KeyError):
                pass
        raise KeyError

    def simextent(self, axis):
        '''
        returns the extent of the actual simulation box.
        Override in your own reader class for better performance implementation.
        '''
        for key in self._simgridkeys():
            try:
                offset = self.gridoffset(key, axis)
                n = self.gridpoints(key, axis)
                return np.array([offset, offset + self.gridspacing(key, axis) * n])
            except(KeyError):
                pass
        raise KeyError('Unable to resolve "simexent" for axis "{:}"'.format(axis))

    def simgridspacing(self, axis):
        extent = self.simextent(axis)
        return (extent[1]-extent[0])/self.simgridpoints(axis)

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

    def __repr__(self):
        return '<Dumpreader at "{:}">'.format(self.dumpidentifier)

    def __eq__(self, other):
        """
        Two dumpreader are equal, if they represent the same dump.

        Assuming both dumpidentifier are paths to the dumpfiles, simple string comparison
        may give a "False",
        although they both point to the same file:
        * ./path/to/file
        * path/to/file
        * /absolute/path/to/file

        Therefore this functions tries to interpret the dumpidentifier as paths/to/files.
        In case this is successful and both files exist,
        the function checks if they point to the same file.
        """
        import os.path as osp
        s1 = str(self.dumpidentifier)
        s2 = str(other.dumpidentifier)
        if osp.isfile(s1) and osp.isfile(s2):
            # osp.samefile available under Windows since python 3.2
            return osp.samefile(s1, s2)
        else:
            # seems to be something else than a path to a file
            return self.dumpidentifier == other.dumpidentifier


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
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            return self._getDumpreader(key)
        else:
            raise TypeError("Invalid argument type.")

    @abc.abstractmethod
    def _getDumpreader(self, number):
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

    def __repr__(self):
        s = '<Simulationreader initialized with "{:}" ({:} dumps)'
        return s.format(self.simidentifier, len(self))

    # Higher Level Functions for usability

    def times(self):
        return np.array([s.time() for s in self])
