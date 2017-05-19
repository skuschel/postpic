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
Dummy reader for creating fake simulation Data for testing purposes.

Stephan Kuschel 2014
'''
from __future__ import absolute_import, division, print_function, unicode_literals

from . import Dumpreader_ifc
from . import Simulationreader_ifc
import numpy as np
from .. import helper
from ..helper import PhysicalConstants


class Dummyreader(Dumpreader_ifc):
    '''
    Dummyreader creates fake Data for testing purposes.

    Args:
      dumpid : int
        the dumpidentifier is the dumpid in this case. It is a float variable,
        that will also change the dummyreaders output (for example it
        will pretend to have dumpid many particles).
    '''

    def __init__(self, dumpid, dimensions=2, randfunc=np.random.normal, seed=0, **kwargs):
        super(self.__class__, self).__init__(dumpid, **kwargs)
        self._dimensions = dimensions
        self._seed = seed
        self._randfunc = randfunc
        # initialize fake data
        if seed is not None:
            np.random.seed(seed)
        self._xdata = randfunc(size=int(dumpid))
        if dimensions > 1:
            self._ydata = randfunc(size=int(dumpid))
        if dimensions > 2:
            self._zdata = randfunc(size=int(dumpid))
        self._pxdata = np.roll(self._xdata, 1) ** 2 * (PhysicalConstants.me * PhysicalConstants.c)
        self._pydata = np.roll(self._xdata, 2) * (PhysicalConstants.me * PhysicalConstants.c)
        self._pzdata = np.roll(self._xdata, 3) * (PhysicalConstants.me * PhysicalConstants.c)
        self._weights = np.repeat(1, len(self._xdata))
        self._ids = np.arange(len(self._xdata))
        np.random.shuffle(self._ids)

    def keys(self):
        pass

    def __getitem__(self, key):
        pass

    def __eq__(self, other):
        ret = super(self.__class__, self).__eq__(other)
        return ret \
            and self._randfunc == other._randfunc \
            and self._seed == other._seed \
            and self._dimensions == other._dimensions

    def timestep(self):
        return self.dumpidentifier

    def time(self):
        return self.timestep() * 1e-10

    def simdimensions(self):
        return self._dimensions

    def gridoffset(self, key, axis):
        raise Exception('Not Implemented')

    def gridspacing(self, key, axis):
        raise Exception('Not Implemented')

    def data(self, axis):
        axid = helper.axesidentify[axis]

        def _Ex(x, y, z):
            ret = np.sin(np.pi * self.timestep() *
                         np.sqrt(x**2 + y**2 + z**2)) / \
                np.sqrt(x**2 + y**2 + z**2 + 1e-9)
            return ret

        def _Ey(x, y, z):
            ret = np.sin(np.pi * self.timestep() * x) + \
                np.sin(np.pi * (self.timestep() - 1) * y) + \
                np.sin(np.pi * (self.timestep() - 2) * z)
            return ret

        def _Ez(x, y, z):
            ret = x**2 + y**2 + z**2
            return ret
        fkts = {0: _Ex,
                1: _Ey,
                2: _Ez}
        if self.simdimensions() == 1:
            ret = fkts[axid](self.grid(None, 'x'), 0, 0)
        elif self.simdimensions() == 2:
            xx, yy = np.meshgrid(self.grid(None, 'x'), self.grid(None, 'y'), indexing='ij')
            ret = fkts[axid](xx, yy, 0)
        elif self.simdimensions() == 3:
            xx, yy, zz = np.meshgrid(self.grid(None, 'x'),
                                     self.grid(None, 'y'),
                                     self.grid(None, 'z'), indexing='ij')
            ret = fkts[axid](xx, yy, zz)
        return ret

    def _keyE(self, component):
        return component

    def _keyB(self, component):
        return component

    def simgridpoints(self, axis):
        return self.grid(None, axis)

    def simextent(self, axis):
        g = self.grid(None, axis)
        return np.asfarray([g[0], g[-1]])

    def gridnode(self, key, axis):
        '''
        Args:
          axis : string or int
            the axisidentifier

        Returns: list of grid points of the axis specified.

        Thus only regular grids are supported currently.
        '''
        axid = helper.axesidentify[axis]
        grids = {1: [(-2, 10, 601)],
                 2: [(-2, 10, 301), (-5, 5, 401)],
                 3: [(-2, 10, 101), (-5, 5, 81), (-4, 4, 61)]}
        if axid >= self.simdimensions():
            raise KeyError('axis ' + str(axis) + ' not present.')
        args = grids[self.simdimensions()][axid]
        ret = np.linspace(*args)
        return ret

    def grid(self, key, axis):
        '''
        Args:
          axis : string or int
            the axisidentifier

        Returns: list of grid points of the axis specified.

        Thus only regular grids are supported currently.
        '''
        axid = helper.axesidentify[axis]
        grids = {1: [(-2, 10, 600)],
                 2: [(-2, 10, 300), (-5, 5, 400)],
                 3: [(-2, 10, 100), (-5, 5, 80), (-4, 4, 60)]}
        if axid >= self.simdimensions():
            raise KeyError('axis ' + str(axis) + ' not present.')
        args = grids[self.simdimensions()][axid]
        ret = np.linspace(*args)
        return ret

    def listSpecies(self):
        return ['electron']

    def getSpecies(self, species, attrib):
        attribid = helper.attribidentify[attrib]
        if attribid == 0:  # x
            ret = self._xdata
        elif attribid == 1 and self.simdimensions() > 1:  # y
            ret = self._ydata
        elif attribid == 2 and self.simdimensions() > 2:  # z
            ret = self._zdata
        elif attribid == 3:  # px
            ret = self._pxdata
        elif attribid == 4:  # py
            ret = self._pydata
        elif attribid == 5:  # pz
            ret = self._pzdata
        elif attribid == 9:  # weights
            ret = self._weights
        elif attribid == 10:  # ids
            ret = self._ids
        else:
            raise KeyError('Attrib "' + str(attrib) + '" of species "' +
                           str(species) + '" not present')
        return ret

    def __str__(self):
        ret = '<Dummyreader ({:d}d) initialized with "' \
            + str(self.dumpidentifier) + '">'
        ret = ret.format(self._dimensions)
        return ret


class Dummysim(Simulationreader_ifc):

    def __init__(self, simidentifier, dimensions=2, **kwargs):
        super(self.__class__, self).__init__(simidentifier, **kwargs)
        self._dimensions = dimensions

    def __len__(self):
        return self.simidentifier

    def _getDumpreader(self, index):
        if index < len(self):
            return Dummyreader(index, dimensions=self._dimensions)
        else:
            raise IndexError()

    def __str__(self):
        ret = '<Dummysimulation ({:d}d) initialized with "' \
            + str(self.dumpidentifier) + '">'
        ret = ret.format(self._dimensions)
        return ret
