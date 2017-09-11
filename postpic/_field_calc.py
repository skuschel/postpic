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
                                        self.gridkeyB('z', **kwargs))
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

    def kspace(self, component, omega_func=None):
        '''
        Reconstruct the physical kspace of one polarization component

        self must be a self object containing the data.
        The required fields will be read from the self.

        component must be one of ["Ex", "Ey", "Ez", "Bx", "By", "Bz"].

        omega_func may be a function that will calculate the dispersion
        relation of the simulation. The function will receive one argument
        that contains the k mesh.

        This function basically computes one component of
            E = 0.5*(E - omega/k^2 * Cross[k, E])
        or
            B = 0.5*(B + 1/omega * Cross[k, B])
        after removing the grid stagger.
        '''
        # target field is polfield and the other field is otherfield
        polfield = component[0]
        otherfield = 'B' if polfield == 'E' else 'E'

        # polarization axis
        polaxis = helper.axesidentify[component[1]]

        # build a dict of all the Field-getter functions of the dump reader
        field_factories = {'E': {0: self.Ex, 1: self.Ey, 2: self.Ez},
                           'B': {0: self.Bx, 1: self.By, 2: self.Bz}}

        # get all the needed fields...
        fields = []

        # the polaxis component of polfield
        fields.append(field_factories[polfield][polaxis]())

        # and the other two components of the otherfield needed for the cross-product
        fields.append(field_factories[otherfield][(polaxis+1) % 3]())
        fields.append(field_factories[otherfield][(polaxis+2) % 3]())

        # find the origin of the simulation box, needed to unstagger the fields
        dims = self.simdimensions()
        simorigin = [self.simextent(i)[0] for i in range(dims)]

        # calculate fourier transform of the fields and unstagger by applying a linear phase
        for field in fields:
            field.fft()
            dx = [so-to for so, to in zip(simorigin, field.transformed_axes_origins)]
            field.shift_grid_by(dx, no_fft=True)

        # start with the target field for the result
        result = fields[0]

        # calculate the k mesh and k^2
        mesh = np.meshgrid(*[ax.grid for ax in result.axes], indexing='ij', sparse=True)
        k2 = sum(ki**2 for ki in mesh)

        # calculate omega, either using the vacuum expression or omega_func()
        if omega_func is None:
            omega = pc.c * np.sqrt(k2)
        else:
            omega = omega_func(mesh)

        # calculate the prefactor in front of the cross product
        # this will produce nan/inf in specific places, which are replaced by 0
        old_settings = np.seterr(all='ignore')
        if polfield == "E":
            prefactor = omega/k2
            prefactor[np.argwhere(np.isnan(prefactor))] = 0.0
        else:
            prefactor = -1.0/omega
            prefactor[np.argwhere(np.isinf(prefactor))] = 0.0
        np.seterr(**old_settings)

        # add/subtract the two terms of the cross-product
        # i chooses the otherfield component  (polaxis+i) % 3
        # mesh_i chooses the k-axis component (polaxis-i) % 3
        # which recreates the crossproduct
        for i in (1, 2):
            mesh_i = (polaxis-i) % 3
            if mesh_i < len(mesh):
                result.matrix += (-1)**(i-1) * prefactor * mesh[mesh_i] * fields[i].matrix

        # divide result by 2 and return
        result.matrix *= 0.5
        return result
