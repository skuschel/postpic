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
# Alexander Blinne, 2017
"""
Field related routines.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from .helper import PhysicalConstants as pc
from . import helper
from .datahandling import *
import warnings

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
        ax = Axis(name=name, unit='m', grid_node=self.gridnode(gridkey, axis))
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
                                        self.gridkeyB('z', **kwargs))
        ret.unit = 'T'
        ret.name = 'Bz'
        ret.shortname = 'Bz'
        return ret

    # --- spezielle Funktionen

    def _kspace(self, component, fields, alignment='auto', solver=None, **kwargs):
        if alignment not in ['auto', 'default', 'epoch', 'epoch-final']:
            raise ValueError()

        if alignment == 'auto':
            if self.name.lower().endswith('.sdf'):
                alignment = 'epoch'
            else:
                alignment = 'default'

        if alignment == 'default':
            return helper.kspace(component, fields, interpolation='fourier', **kwargs)

        if alignment.startswith('epoch'):
            dt = self.time()/self.timestep()
            if 'omega_func' not in kwargs and solver == 'yee':
                dx = [self.simgridspacing(axis) for axis in range(self.simdimensions())]
                kwargs['omega_func'] = helper.omega_yee_factory(dx, dt)

        if alignment == 'epoch':
            return helper.kspace_epoch_like(component, fields, dt, align_to='B', **kwargs)

        if alignment == 'epoch-final':
            return helper.kspace_epoch_like(component, fields, dt, align_to='E', **kwargs)

    def kspace_Ex(self, **kwargs):
        fields = dict()
        fields['Ex'] = self.Ex()
        if fields['Ex'].dimensions >= 2:
            fields['Bz'] = self.Bz()
        if fields['Ex'].dimensions >= 3:
            fields['By'] = self.By()

        return self._kspace('Ex', fields, **kwargs)

    def kspace_Ey(self, **kwargs):
        fields = dict()
        fields['Ey'] = self.Ey()
        fields['Bz'] = self.Bz()
        if fields['Ey'].dimensions >= 3:
            fields['Bx'] = self.Bx()

        return self._kspace('Ey', fields, **kwargs)

    def kspace_Ez(self, **kwargs):
        fields = dict()
        fields['Ez'] = self.Ez()
        fields['By'] = self.By()
        if fields['Ez'].dimensions >= 2:
            fields['Bx'] = self.Bx()

        return self._kspace('Ez', fields, **kwargs)

    def kspace_Bx(self, **kwargs):
        fields = dict()
        fields['Bx'] = self.Bx()
        if fields['Bx'].dimensions >= 2:
            fields['Ez'] = self.Ez()
        if fields['Bx'].dimensions >= 3:
            fields['Ey'] = self.Ey()

        return self._kspace('Bx', fields, **kwargs)

    def kspace_By(self, **kwargs):
        fields = dict()
        fields['By'] = self.By()
        fields['Ez'] = self.Ez()
        if fields['By'].dimensions >= 3:
            fields['Ex'] = self.Ex()

        return self._kspace('By', fields, **kwargs)

    def kspace_Bz(self, **kwargs):
        fields = dict()
        fields['Bz'] = self.Bz()
        fields['Ey'] = self.Ey()
        if fields['Bz'].dimensions >= 2:
            fields['Ex'] = self.Ex()

        return self._kspace('Bz', fields, **kwargs)

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

    def _divE1d(self, **kwargs):
        return np.gradient(self._Ex(**kwargs))

    def _divE2d(self, **kwargs):
        from pkg_resources import parse_version
        if parse_version(np.__version__) < parse_version('1.11'):
            warnings.warn('''
            The support for numpy < "1.11" will be dropped in the future. Upgrade!
            ''', DeprecationWarning)
            return np.gradient(self._Ex(**kwargs))[0] \
                + np.gradient(self._Ey(**kwargs))[1]
        return np.gradient(self._Ex(**kwargs), axis=0) \
            + np.gradient(self._Ey(**kwargs), axis=1)

    def _divE3d(self, **kwargs):
        from pkg_resources import parse_version
        if parse_version(np.__version__) < parse_version('1.11'):
            warnings.warn('''
            The support for numpy < "1.11" will be dropped in the future. Upgrade!
            ''', DeprecationWarning)
            return np.gradient(self._Ex(**kwargs))[0] \
                + np.gradient(self._Ey(**kwargs))[1] \
                + np.gradient(self._Ez(**kwargs))[2]
        return np.gradient(self._Ex(**kwargs), axis=0) \
            + np.gradient(self._Ey(**kwargs), axis=1) \
            + np.gradient(self._Ez(**kwargs), axis=2)

    def divE(self, **kwargs):
        '''
        returns the divergence of E.
        This is calculated in the number of dimensions the simulation was running on.
        '''
        # this works because the datareader extents this class
        simdims = self.simdimensions()
        opts = {1: self._divE1d,
                2: self._divE2d,
                3: self._divE3d}
        data = opts[simdims](**kwargs)
        ret = self._createfieldfromdata(data, self.gridkeyE('x', **kwargs))
        ret.unit = 'V/m^2'
        ret.name = 'div E'
        ret.shortname = 'divE'
        return ret
