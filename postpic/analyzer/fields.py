"""
Field related routines.
"""

__all__ = ['FieldAnalyzer']

import numpy as np
import analyzer.PhysicalConstants as pc



class FieldAnalyzer(_Constants):


    def __init__(self, sdfanalyzer, lasnm=None):
        self.sdfanalyzer = sdfanalyzer
        self.lasnm = lasnm
        if lasnm:
            self.k0 = 2 * np.pi / (lasnm * 1e-9)
        else:
            self.k0 = None
        self.simdimensions = sdfanalyzer.simdimensions
        self._simextent = sdfanalyzer.simextent()
        self._simgridpoints = sdfanalyzer.simgridpoints
        self._extent = self._simextent.copy()  # Variable definiert den ausgeschnittenen Bereich

    def datenausschnitt_bound(self, m):
        if all(self._simextent == self._extent):
            ret = m
        else:
            ret = self.datenausschnitt(m, self._simextent, self._extent)
        return ret

    def getsimextent(self, axis=None):
        return self._simextent.copy()[self._axisoptions[axis]]

    def getextent(self, axis=None):
        return self._extent.copy()[self._axisoptions[axis]]

    def getsimgridpoints(self, axis=None):
        return self._simgridpoints.copy()[self._axisoptionseinzel[axis]]

    def getsimdomainsize(self, axis=None):
        return np.diff(self.getsimextent(axis))[0::2]

    def getspatialresolution(self, axis=None):
        return self.getsimdomainsize() / self.getsimgridpoints()

    def setextent(self, newextent, axis=None):
        self._extent[self._axisoptions[axis]] = newextent

    def setspacialtofield(self, field):
        """
        Fuegt dem Feld alle Informationen uber das rauemliche Grid hinzu.
        """
        field.setallaxesspacial()
        field.setgrid_node(0, self.sdfanalyzer.grid_node('x'))
        if self.simdimensions > 1:
            field.setgrid_node(1, self.sdfanalyzer.grid_node('y'))
        if self.simdimensions > 2:
            field.setgrid_node(2, self.sdfanalyzer.grid_node('z'))
        return None


    # --- Return functions for basic data layer

    # -- basic --
    # **kwargs ist z.B. average=True
    def _Ex(self, **kwargs):
        return self.datenausschnitt_bound(self.sdfanalyzer.dataE('x', **kwargs))
    def _Ey(self, **kwargs):
        return self.datenausschnitt_bound(self.sdfanalyzer.dataE('y', **kwargs))
    def _Ez(self, **kwargs):
        return self.datenausschnitt_bound(self.sdfanalyzer.dataE('z', **kwargs))
    def _Bx(self, **kwargs):
        return self.datenausschnitt_bound(self.sdfanalyzer.dataB('x', **kwargs))
    def _By(self, **kwargs):
        return self.datenausschnitt_bound(self.sdfanalyzer.dataB('y', **kwargs))
    def _Bz(self, **kwargs):
        return self.datenausschnitt_bound(self.sdfanalyzer.dataB('z', **kwargs))


    # --- Alle Funktionen geben ein Objekt vom Typ Feld zurueck

    # allgemein ueber dem Ort auftragen. Insbesondere fuer Derived/*
    def createfeldfromkey(self, key):
        ret = Feld(self.datenausschnitt_bound(self.sdfanalyzer.data(key)));
        ret.name = key
        self.setspacialtofield(ret)
        return ret

    def createfelderfromkeys(self, *keys):
        ret = ()
        for key in keys:
            ret += (self.createfeldfromkey(key),)
        return ret


    # jetzt alle einzeln
    def Ex(self, **kwargs):
        ret = Feld(self._Ex(**kwargs))
        ret.unit = 'V/m'
        ret.name = 'Ex'
        ret.label = 'Ex'
        self.setspacialtofield(ret)
        return ret

    def Ey(self, **kwargs):
        ret = Feld(self._Ey(**kwargs))
        ret.unit = 'V/m'
        ret.name = 'Ey'
        ret.label = 'Ey'
        self.setspacialtofield(ret)
        return ret

    def Ez(self, **kwargs):
        ret = Feld(self._Ez(**kwargs))
        ret.unit = 'V/m'
        ret.name = 'Ez'
        ret.label = 'Ez'
        self.setspacialtofield(ret)
        return ret

    def Bx(self, **kwargs):
        ret = Feld(self._Bx(**kwargs))
        ret.unit = 'T'
        ret.name = 'Bx'
        ret.label = 'Bx'
        self.setspacialtofield(ret)
        return ret

    def By(self, **kwargs):
        ret = Feld(self._By(**kwargs))
        ret.unit = 'T'
        ret.name = 'By'
        ret.label = 'By'
        self.setspacialtofield(ret)
        return ret

    def Bz(self, **kwargs):
        ret = Feld(self._Bz(**kwargs))
        ret.unit = 'T'
        ret.name = 'Bz'
        ret.label = 'Bz'
        self.setspacialtofield(ret)
        return ret



    # --- spezielle Funktionen

    def energydensityE(self, **kwargs):
        ret = Feld(0.5 * self._epsilon0 * (self._Ex(**kwargs) ** 2 + self._Ey(**kwargs) ** 2 + self._Ez(**kwargs) ** 2))
        ret.unit = 'J/m^3'
        ret.name = 'Energy Density Electric-Field'
        ret.label = 'E'
        self.setspacialtofield(ret)
        return ret

    def energydensityM(self, **kwargs):
        ret = Feld(0.5 / self._mu0 * (self._Bx(**kwargs) ** 2 + self._By(**kwargs) ** 2 + self._Bz(**kwargs) ** 2))
        ret.unit = 'J/m^3'
        ret.name = 'Energy Density Magnetic-Field'
        ret.label = 'M'
        self.setspacialtofield(ret)
        return ret

    def energydensityEM(self, **kwargs):
        ret = Feld(0.5 * self._epsilon0 * (self._Ex(**kwargs) ** 2 + self._Ey(**kwargs) ** 2 + self._Ez(**kwargs) ** 2) \
             + 0.5 / self._mu0 * (self._Bx(**kwargs) ** 2 + self._By(**kwargs) ** 2 + self._Bz(**kwargs) ** 2))
        ret.unit = 'J/m^3'
        ret.name = 'Energy Density EM-Field'
        ret.label = 'EM'
        self.setspacialtofield(ret)
        return ret

    # --- Spektren

    def spectrumEx(self, axis=0):
        if self.k0 == None:
            ret = Feld(None)
            print 'WARNING: lasnm not given. Spectrum will not be calculated.'
        else:
            rfftaxes = np.roll((0, 1), axis)
            ret = Feld(0.5 * self._epsilon0 * abs(np.fft.fftshift(np.fft.rfft2(self._Ex(), axes=rfftaxes), axes=axis)) ** 2)
        ret.unit = '?'
        ret.name = 'Spectrum Ex'
        ret.label = 'Spectrum Ex'
        ret.setallaxes(name=[r'$k_x$', r'$k_y$', r'$k_z$'], unit=['', '', ''])
        extent = np.zeros(2 * self.simdimensions)
        extent[1::2] = np.pi / self.getspatialresolution()
        if self.k0:
            ret.setallaxes(name=[r'$k_x / k_0$', r'$k_y / k_0$', r'$k_z / k_0$'], unit=['$\lambda_0 =$' + str(self.lasnm) + 'nm', '', ''])
            extent[1::2] = extent[1::2] / self.k0
        mittel = np.mean(extent[(0 + 2 * axis):(2 + 2 * axis)])
        extent[0 + 2 * axis] = -2 * mittel
        extent[1 + 2 * axis] = 2 * mittel
        ret.setgrid_node_fromextent(extent)
        return ret

    def spectrumBz(self, axis=0):
        if self.k0 == None:
            ret = Feld(None)
            print 'WARNING: lasnm not given. Spectrum will not be calculated.'
        else:
            rfftaxes = np.roll((0, 1), axis)
            ret = Feld(0.5 / self._mu0 * abs(np.fft.fftshift(np.fft.rfft2(self._Bz(), axes=rfftaxes), axes=axis)) ** 2)
        ret.unit = '?'
        ret.name = 'Spectrum Bz'
        ret.label = 'Spectrum Bz'
        ret.setallaxes(name=[r'$k_x$', r'$k_y$', r'$k_z$'], unit=['', '', ''])
        extent = np.zeros(2 * self.simdimensions)
        extent[1::2] = np.pi / self.getspatialresolution()
        if self.k0:
            ret.setallaxes(name=[r'$k_x / k_0$', r'$k_y / k_0$', r'$k_z / k_0$'], unit=['$\lambda_0 =$' + str(self.lasnm) + 'nm', '', ''])
            extent[1::2] = extent[1::2] / self.k0
        mittel = np.mean(extent[(0 + 2 * axis):(2 + 2 * axis)])
        extent[0 + 2 * axis] = -2 * mittel
        extent[1 + 2 * axis] = 2 * mittel
        ret.setgrid_node_fromextent(extent)
        return ret




