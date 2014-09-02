"""
Field related routines.
"""

__all__ = ['FieldAnalyzer']

import numpy as np
from analyzer import PhysicalConstants as pc
from ..datahandling import *


class FieldAnalyzer(object):
    '''
    This class transforms any data written to a dump to a field object
    ready to be plotted. This should provide an object to make it easy
    to plot data, that is already dumped.

    Simple calucations (like calculating the energy density) are
    performed here as well but might move to another class in the future.

    Entirely new calculations (like fourier transforming gridded data)
    will be build up somewhere else, since they act on a
    field transforming it into another.
    '''

    def __init__(self, dumpreader):
        self.dumpreader = dumpreader

    def setspacialtofield(self, field):
        '''
        add spacial field information to the given field object.
        '''
        field.setaxisobj('x', self.dumpreader.getaxis('x'))
        if self.dumpreader.simdimensions() > 1:
            field.setaxisobj('y', self.dumpreader.getaxis('y'))
        if self.dumpreader.simdimensions() > 2:
            field.setaxisobj('z', self.dumpreader.getaxis('z'))

    # --- Return functions for basic data layer
    @staticmethod
    def _returnfunc(data):
        return np.float64(data)

    # -- basic --
    # **kwargs is not used up to now
    def _Ex(self, **kwargs):
        return self._returnfunc(self.dumpreader.dataE('x', **kwargs))

    def _Ey(self, **kwargs):
        return self._returnfunc(self.dumpreader.dataE('y', **kwargs))

    def _Ez(self, **kwargs):
        return self._returnfunc(self.dumpreader.dataE('z', **kwargs))

    def _Bx(self, **kwargs):
        return self._returnfunc(self.dumpreader.dataB('x', **kwargs))

    def _By(self, **kwargs):
        return self._returnfunc(self.dumpreader.dataB('y', **kwargs))

    def _Bz(self, **kwargs):
        return self._returnfunc(self.dumpreader.dataB('z', **kwargs))

    # --- Always return an object of Field type

    # General interface for everything
    def createfieldfromkey(self, key):
        ret = Field(self._returnfunc(self.dumpreader[key]))
        ret.name = key
        self.setspacialtofield(ret)
        return ret

    def createfieldsfromkeys(self, *keys):
        for key in keys:
            yield self.createfieldfromkey(key)

    # most common fields listed here nicely
    def Ex(self, **kwargs):
        ret = Field(self._Ex(**kwargs))
        ret.unit = 'V/m'
        ret.name = 'Ex'
        ret.shortname = 'Ex'
        self.setspacialtofield(ret)
        return ret

    def Ey(self, **kwargs):
        ret = Field(self._Ey(**kwargs))
        ret.unit = 'V/m'
        ret.name = 'Ey'
        ret.shortname = 'Ey'
        self.setspacialtofield(ret)
        return ret

    def Ez(self, **kwargs):
        ret = Field(self._Ez(**kwargs))
        ret.unit = 'V/m'
        ret.name = 'Ez'
        ret.shortname = 'Ez'
        self.setspacialtofield(ret)
        return ret

    def Bx(self, **kwargs):
        ret = Field(self._Bx(**kwargs))
        ret.unit = 'T'
        ret.name = 'Bx'
        ret.shortname = 'Bx'
        self.setspacialtofield(ret)
        return ret

    def By(self, **kwargs):
        ret = Field(self._By(**kwargs))
        ret.unit = 'T'
        ret.name = 'By'
        ret.shortname = 'By'
        self.setspacialtofield(ret)
        return ret

    def Bz(self, **kwargs):
        ret = Field(self._Bz(**kwargs))
        ret.unit = 'T'
        ret.name = 'Bz'
        ret.shortname = 'Bz'
        self.setspacialtofield(ret)
        return ret

    # --- spezielle Funktionen

    def energydensityE(self, **kwargs):
        ret = Field(0.5 * pc.epsilon0
                    * (self._Ex(**kwargs) ** 2
                       + self._Ey(**kwargs) ** 2
                       + self._Ez(**kwargs) ** 2))
        ret.unit = 'J/m^3'
        ret.name = 'Energy Density Electric-Field'
        ret.shortname = 'E'
        self.setspacialtofield(ret)
        return ret

    def energydensityM(self, **kwargs):
        ret = Field(0.5 / pc.mu0
                    * (self._Bx(**kwargs) ** 2
                       + self._By(**kwargs) ** 2
                       + self._Bz(**kwargs) ** 2))
        ret.unit = 'J/m^3'
        ret.name = 'Energy Density Magnetic-Field'
        ret.shortname = 'M'
        self.setspacialtofield(ret)
        return ret

    def energydensityEM(self, **kwargs):
        ret = Field(0.5 * pc.epsilon0
                    * (self._Ex(**kwargs) ** 2
                       + self._Ey(**kwargs) ** 2
                       + self._Ez(**kwargs) ** 2)
                    + 0.5 / pc.mu0
                    * (self._Bx(**kwargs) ** 2
                       + self._By(**kwargs) ** 2
                       + self._Bz(**kwargs) ** 2))
        ret.unit = 'J/m^3'
        ret.name = 'Energy Density EM-Field'
        ret.shortname = 'EM'
        self.setspacialtofield(ret)
        return ret

