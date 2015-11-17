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
"""
Field related routines.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from .helper import PhysicalConstants as pc
from . import helper
from .datahandling import *

__all__ = ['FieldAnalyzer']


class FieldAnalyzer(object):
    '''
    This class transforms any data written to a dump to a field object
    ready to be plotted. This should provide an object to make it easy
    to plot data, that is already dumped.

    Since the postpic.datareader.Dumpreader_ifc is derived from this class,
    all methods here are automatically available to all dumpreaders.

    Simple calucations (like calculating the energy density) are
    performed here as well but might move to another class in the future.

    Entirely new calculations (like fourier transforming gridded data)
    will be build up somewhere else, since they act on a
    field transforming it into another.
    '''

    def __init__(self):
        pass

    # General interface for everything
    def _createfieldfromdata(self, data, gridkey):
        ret = Field(np.float64(data))
        self.setgridtofield(ret, gridkey)
        return ret

    def createfieldfromkey(self, key, gridkey=None):
        '''
        This method creates a Field object from the data identified by "key".
        The Grid is also inferred from that key unless an alternate "gridkey"
        is provided.
        '''
        if gridkey is None:
            gridkey = key
        ret = self._createfieldfromdata(self.data(key), gridkey)
        ret.name = key
        return ret

    def getaxisobj(self, gridkey, axis):
        '''
        returns an Axis object for the "axis" and the grid defined by "gridkey".
        '''
        name = {0: 'x', 1: 'y', 2: 'z'}[helper.axesidentify[axis]]
        ax = Axis(name=name, unit='m')
        ax.grid_node = self.gridnode(gridkey, axis)
        return ax

    def setgridtofield(self, field, gridkey):
        '''
        add spacial field information to the given field object.
        '''
        field.setaxisobj('x', self.getaxisobj(gridkey, 'x'))
        if field.dimensions > 1:
            field.setaxisobj('y', self.getaxisobj(gridkey, 'y'))
        if field.dimensions > 2:
            field.setaxisobj('z', self.getaxisobj(gridkey, 'z'))

    # --- Always return an object of Field type
    # just to shortcut

    def _Ex(self, **kwargs):
        return np.float64(self.dataE('x', **kwargs))

    def _Ey(self, **kwargs):
        return np.float64(self.dataE('y', **kwargs))

    def _Ez(self, **kwargs):
        return np.float64(self.dataE('z', **kwargs))

    def _Bx(self, **kwargs):
        return np.float64(self.dataB('x', **kwargs))

    def _By(self, **kwargs):
        return np.float64(self.dataB('y', **kwargs))

    def _Bz(self, **kwargs):
        return np.float64(self.dataB('z', **kwargs))

    def createfieldsfromkeys(self, *keys):
        for key in keys:
            yield self.createfieldfromkey(key)

    # most common fields listed here nicely
    def Ex(self, **kwargs):
        ret = self._createfieldfromdata(self._Ex(**kwargs),
                                        self.gridkeyE('x', **kwargs))
        ret.unit = 'V/m'
        ret.name = 'Ex'
        ret.shortname = 'Ex'
        return ret

    def Ey(self, **kwargs):
        ret = self._createfieldfromdata(self._Ey(**kwargs),
                                        self.gridkeyE('y', **kwargs))
        ret.unit = 'V/m'
        ret.name = 'Ey'
        ret.shortname = 'Ey'
        return ret

    def Ez(self, **kwargs):
        ret = self._createfieldfromdata(self._Ez(**kwargs),
                                        self.gridkeyE('z', **kwargs))
        ret.unit = 'V/m'
        ret.name = 'Ez'
        ret.shortname = 'Ez'
        return ret

    def Bx(self, **kwargs):
        ret = self._createfieldfromdata(self._Bx(**kwargs),
                                        self.gridkeyB('x', **kwargs))
        ret.unit = 'T'
        ret.name = 'Bx'
        ret.shortname = 'Bx'
        return ret

    def By(self, **kwargs):
        ret = self._createfieldfromdata(self._By(**kwargs),
                                        self.gridkeyB('y', **kwargs))
        ret.unit = 'T'
        ret.name = 'By'
        ret.shortname = 'By'
        return ret

    def Bz(self, **kwargs):
        ret = self._createfieldfromdata(self._Bz(**kwargs),
                                        self.gridkeyB('y', **kwargs))
        ret.unit = 'T'
        ret.name = 'Bz'
        ret.shortname = 'Bz'
        return ret

    # --- spezielle Funktionen

    def energydensityE(self, **kwargs):
        ret = self._createfieldfromdata(0.5 * pc.epsilon0 *
                                        (self._Ex(**kwargs) ** 2 +
                                         self._Ey(**kwargs) ** 2 +
                                         self._Ez(**kwargs) ** 2),
                                        self.gridkeyE('x', **kwargs))
        ret.unit = 'J/m^3'
        ret.name = 'Energy Density Electric-Field'
        ret.shortname = 'E'
        return ret

    def energydensityM(self, **kwargs):
        ret = self._createfieldfromdata(0.5 / pc.mu0 *
                                        (self._Bx(**kwargs) ** 2 +
                                         self._By(**kwargs) ** 2 +
                                         self._Bz(**kwargs) ** 2),
                                        self.gridkeyB('x', **kwargs))
        ret.unit = 'J/m^3'
        ret.name = 'Energy Density Magnetic-Field'
        ret.shortname = 'M'
        return ret

    def energydensityEM(self, **kwargs):
        ret = self._createfieldfromdata(0.5 * pc.epsilon0 *
                                        (self._Ex(**kwargs) ** 2 +
                                         self._Ey(**kwargs) ** 2 +
                                         self._Ez(**kwargs) ** 2) +
                                        0.5 / pc.mu0 *
                                        (self._Bx(**kwargs) ** 2 +
                                         self._By(**kwargs) ** 2 +
                                         self._Bz(**kwargs) ** 2),
                                        self.gridkeyE('x', **kwargs))
        ret.unit = 'J/m^3'
        ret.name = 'Energy Density EM-Field'
        ret.shortname = 'EM'
        return ret

