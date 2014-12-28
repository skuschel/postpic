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
Particle related routines.
"""

__all__ = ['ParticleAnalyzer']

import numpy as np
from analyzer import PhysicalConstants as pc
import analyzer
from ..datahandling import *

identifyspecies = analyzer.SpeciesIdentifier.identifyspecies


class _SingleSpeciesAnalyzer(object):
    """
    used by the ParticleAnalyzer class only.
    """

    def __init__(self, dumpreader, species):
        self.species = species
        self._dumpreader = dumpreader
        try:
            self._mass = dumpreader.getSpecies(species, 'mass')
            self._charge = dumpreader.getSpecies(species, 'charge')
        except(KeyError):
            self._idfy = identifyspecies(species)
            self._mass = self._idfy['mass']  # SI
            self._charge = self._idfy['charge']  # SI
        self.compresslog = []
        self._compressboollist = None
        self._cache = {}
        # Variables will be read and added to cache when needed.

    def __getitem__(self, key):
        '''
        Reads a key property, thus one out of
        weight, x, y, z, px, py, pz, ID
        '''
        if key in self._cache:
            ret = self._cache[key]
        else:
            ret = self._dumpreader.getSpecies(self.species, key)
            ret = np.float64(ret)
            if not isinstance(ret, float) and self._compressboollist is not None:
                ret = ret[self._compressboollist]  # avoid executing this line too often.
                self._cache[key] = ret
                # if memomry is low, caching could be skipped entirely.
                # See commit message for benchmark.
        return ret

    def compress(self, condition, name='unknown condition'):
        """
        works like numpy.compress.
        Additionaly you can specify a name, that gets saved in the compresslog.

        condition has to be one out of:
        1)
        condition =  [True, False, True, True, ... , True, False]
        condition is a list of length N, specifing which particles to keep.
        Example:
        cfintospectrometer = lambda x: x.angle_offaxis() < 30e-3
        cfintospectrometer.name = '< 30mrad offaxis'
        pa.compress(cfintospectrometer(pa), name=cfintospectrometer.name)
        2)
        condtition = [1, 2, 4, 5, 9, ... , 805, 809]
        condition can be a list of arbitraty length, so only the particles
        with the ids listed here are kept.
        """
        if np.array(condition).dtype is np.dtype('bool'):
            # Case 1:
            # condition is list of boolean values specifying particles to use
            assert len(self) == len(condition), \
                'number of particles ({:7n}) has to match' \
                'length of condition ({:7n})' \
                ''.format(len(self), len(condition))
            if self._compressboollist is None:
                self._compressboollist = condition
            else:
                self._compressboollist[self._compressboollist] = condition
            for key in self._cache:
                self._cache[key] = self._cache[key][condition]
            self.compresslog = np.append(self.compresslog, name)
        else:
            # Case 2:
            # condition is list of particle IDs to use
            condition = np.array(condition, dtype='int')
            # same as
            # bools = np.array([idx in condition for idx in self._ID])
            # but benchmarked to be 1500 times faster :)
            condition.sort()
            idx = np.searchsorted(condition, self._ID)
            idx[idx == len(condition)] = 0
            bools = condition[idx] == self._ID
            self.compress(bools, name=name)

    def uncompress(self):
        """
        Discard all previous runs of 'compress'
        """
        self.compresslog = []
        self._compressboollist = None
        self._cache = {}

    # --- Only very basic functions

    def __len__(self):  # = number of particles
        # find a valid dataset to count number of paricles
        if self._compressboollist is not None:
            return len(self._compressboollist)
        for key in ['weight', 'x', 'px', 'y', 'py', 'z', 'pz']:
            data = self[key]
            try:
                # len(3) will yield a TypeError, len([3]) returns 1
                ret = len(data)
                break
            except(TypeError):
                pass
        return ret

# --- These functions are for practical use. Return None if not dumped.

    def weight(self):  # np.float64(np.array([4.3])) == 4.3 may cause error
        w = self['weight']
        # on constant weight, self['weight'] may return a single scalar,
        # which is converted to float by __getitem__
        if isinstance(w, float):
            ret = np.repeat(w, len(self))
        else:
            ret = w
        return ret

    def mass(self):  # SI
        return np.repeat(self._mass, len(self))

    def charge(self):  # SI
        return np.repeat(self._charge, len(self))

    def Px(self):
        return self['px']

    def Py(self):
        return self['py']

    def Pz(self):
        return self['pz']

    def X(self):
        return self['x']

    def Y(self):
        return self['y']

    def Z(self):
        return self['z']

    def ID(self):
        ret = self['ID']
        if ret is not None:
            ret = np.array(ret, dtype=int)
        return ret


class ParticleAnalyzer(object):
    """
    The ParticleAnalyzer class. Different ParticleAnalyzer can be
    added together to create a combined collection.
    """

    def __init__(self, dumpreader, *speciess):
        # create 'empty' ParticleAnalyzer
        self._ssas = []
        self._species = None  # trivial name if set
        self._compresslog = []
        self.simdimensions = dumpreader.simdimensions()
        try:
            self.X.__func__.extent = dumpreader.extent('x')
            self.X.__func__.gridpoints = dumpreader.gridpoints('x')
            self.X_um.__func__.extent = dumpreader.extent('x') * 1e6
            self.X_um.__func__.gridpoints = dumpreader.gridpoints('x')
            self.Y.__func__.extent = dumpreader.extent('y')
            self.Y.__func__.gridpoints = dumpreader.gridpoints('y')
            self.Y_um.__func__.extent = dumpreader.extent('y') * 1e6
            self.Y_um.__func__.gridpoints = dumpreader.gridpoints('y')
            self.Z.__func__.extent = dumpreader.extent('z')
            self.Z.__func__.gridpoints = dumpreader.gridpoints('z')
            self.Z_um.__func__.extent = dumpreader.extent('z') * 1e6
            self.Z_um.__func__.gridpoints = dumpreader.gridpoints('z')
        except(KeyError):
            pass
        self.angle_xy.__func__.extent = np.real([-np.pi, np.pi])
        self.angle_yz.__func__.extent = np.real([-np.pi, np.pi])
        self.angle_zx.__func__.extent = np.real([-np.pi, np.pi])
        # add particle species one by one
        for s in speciess:
            self.add(dumpreader, s)

    def __str__(self):
        return '<ParticleAnalyzer including ' + str(self.species) \
            + '(' + str(len(self)) + ')>'

    @property
    def npart(self):
        '''
        Number of Particles.
        '''
        ret = 0
        for ssa in self._ssas:
            ret += len(ssa)
        return ret

    @property
    def nspecies(self):
        return len(self._ssas)

    def __len__(self):
        return self.npart

    @property
    def species(self):
        '''
        returns an string name for the species involved.
        Basically only returns unique names from all species
        (used for plotting and labeling purposes -- not for completeness).
        May be overwritten.
        '''
        if self._species is not None:
            return self._species
        ret = ''
        for s in set(self.speciess):
            ret += s + ' '
        ret = ret[0:-1]
        return ret

    @species.setter
    def species(self, name):
        self._name = name

    @property
    def name(self):
        '''
        an alias to self.species
        '''
        return self.species

    @property
    def speciess(self):
        '''
        a complete list of all species involved.
        '''
        return [ssa.species for ssa in self._ssas]

    def add(self, dumpreader, species):
        '''
        adds species to this analyzer.

        Attributes
        ----------
        species can be a single species name
                or a reserved name for collection of species, such as
                ions    adds all available particles that are ions
                nonions adds all available particles that are not ions
                ejected
                noejected
                all
        '''
        keys = {'ions': lambda s: identifyspecies(s)['ision'],
                'nonions': lambda s: not identifyspecies(s)['ision'],
                'ejected': lambda s: identifyspecies(s)['ejected'],
                'noejected': lambda s: not identifyspecies(s)['ejected'],
                'all': lambda s: True}
        if species in keys:
            ls = dumpreader.listSpecies()
            toadd = [s for s in ls if keys[species](s)]
            for s in toadd:
                self.add(dumpreader, s)
        else:
            self._ssas.append(_SingleSpeciesAnalyzer(dumpreader, species))
        return

    # --- Operator overloading

    def __add__(self, other):  # self + other
        ret = copy.copy(self)
        ret += other
        return ret

    def __iadd__(self, other):  # self += other
        '''
        adding ParticleAnalyzers should give the feeling as if you were adding
        their particle lists. Thats why there is no append function.
        Compare those outputs (numpy.array handles that differently!):
        a=[1,2,3]; a.append([4,5]); print a
        [1,2,3,[4,5]]
        a=[1,2,3]; a += [4,5]; print a
        [1,2,3,4,5]
        '''
        # only add ssa with more than 0 particles.
        for ssa in other._ssas:
            if len(ssa) > 0:
                self._ssas.append(copy.copy(ssa))
        return self

    # --- only point BASIC functions to SingleSpeciesAnalyzer

    def _funcabbilden(self, func):
        ret = np.array([])
        for ssa in self._ssas:
            a = getattr(ssa, func)()
            if a is None:
                # This particle property is not dumped in the
                # current SingleSpeciesAnalyzer
                continue
            if len(a) > 0:
                ret = np.append(ret, a)
        return ret

    def _weight(self):
        return self._funcabbilden('weight')

    def _mass(self):  # SI
        return self._funcabbilden('mass')

    def _charge(self):  # SI
        return self._funcabbilden('charge')

    def _Px(self):
        return self._funcabbilden('Px')

    def _Py(self):
        return self._funcabbilden('Py')

    def _Pz(self):
        return self._funcabbilden('Pz')

    def _X(self):
        return self._funcabbilden('X')

    def _Y(self):
        return self._funcabbilden('Y')

    def _Z(self):
        return self._funcabbilden('Z')

    def ID(self):
        return self._funcabbilden('ID')

    def compress(self, condition, name='unknown condition'):
        i = 0
        for ssa in self._ssas:  # condition is list of booleans
            if condition.dtype == np.dtype('bool'):
                n = ssa.weight().shape[0]
                ssa.compress(condition[i:i + n], name=name)
                i += n
            else:  # condition is list of particle IDs
                ssa.compress(condition, name=name)
        self._compresslog = np.append(self._compresslog, name)

    # --- user friendly functions

    def compressfn(self, conditionf, name='unknown condition'):
        if hasattr(conditionf, 'name'):
            name = conditionf.name
        self.compress(conditionf(self), name=name)

    def uncompress(self):
        self._compresslog = []
        for s in self._ssas:
            s.uncompress()
        # self.__init__(self.data, *self._speciess)

    def _mass_u(self):
        return self._mass() / pc.mass_u

    def _charge_e(self):
        return self._charge() / pc.qe

    def _Eruhe(self):
        return self._mass() * pc.c ** 2

    def getcompresslog(self):
        ret = {'all': self._compresslog}
        for ssa in self._ssas:
            ret.update({ssa.species: ssa.compresslog})
        return ret

    # --- "A scalar for every particle"-functions.

    def weight(self):
        return self._weight()
    weight.name = 'Particle weight'
    weight.unit = ''

    def Px(self):
        return self._Px()
    Px.unit = ''
    Px.name = 'Px'

    def Py(self):
        return self._Py()
    Py.unit = ''
    Py.name = 'Py'

    def Pz(self):
        return self._Pz()
    Pz.unit = ''
    Pz.name = 'Pz'

    def P(self):
        return np.sqrt(self._Px() ** 2 + self._Py() ** 2 + self._Pz() ** 2)
    P.unit = ''
    P.name = 'P'

    def X(self):
        return self._X()
    X.unit = 'm'
    X.name = 'X'

    def X_um(self):
        return self._X() * 1e6
    X_um.unit = '$\mu m$'
    X_um.name = 'X'

    def Y(self):
        return self._Y()
    Y.unit = 'm'
    Y.name = 'Y'

    def Y_um(self):
        return self._Y() * 1e6
    Y_um.unit = '$\mu m$'
    Y_um.name = 'Y'

    def Z(self):
        return self._Z()
    Z.unit = 'm'
    Z.name = 'Z'

    def Z_um(self):
        return self._Z() * 1e6
    Z_um.unit = '$\mu m$'
    Z_um.name = 'Z'

    def beta(self):
        return np.sqrt(self.gamma() ** 2 - 1) / self.gamma()
    beta.unit = r'$\beta$'
    beta.name = 'beta'

    def V(self):
        return pc.c * self.beta()
    V.unit = 'm/s'
    V.name = 'V'

    def gamma(self):
        return np.sqrt(1 +
                       (self._Px() ** 2 + self._Py() ** 2 + self._Pz() ** 2)
                       / (self._mass() * pc.c) ** 2)
    gamma.unit = r'$\gamma$'
    gamma.name = 'gamma'

    def Ekin(self):
        return (self.gamma() - 1) * self._Eruhe()
    Ekin.unit = 'J'
    Ekin.name = 'Ekin'

    def Ekin_MeV(self):
        return self.Ekin() / pc.qe / 1e6
    Ekin_MeV.unit = 'MeV'
    Ekin_MeV.name = 'Ekin'

    def Ekin_MeV_amu(self):
        return self.Ekin_MeV() / self._mass_u()
    Ekin_MeV_amu.unit = 'MeV / amu'
    Ekin_MeV_amu.name = 'Ekin / amu'

    def Ekin_MeV_qm(self):
        return self.Ekin_MeV() * self._charge_e() / self._mass_u()
    Ekin_MeV_qm.unit = 'MeV*q/m'
    Ekin_MeV_qm.name = 'Ekin * q/m'

    def Ekin_keV(self):
        return self.Ekin() / pc.qe / 1e3
    Ekin_keV.unit = 'keV'
    Ekin_keV.name = 'Ekin'

    def Ekin_keV_amu(self):
        return self.Ekin_keV() / self._mass_u()
    Ekin_keV_amu.unit = 'keV / amu'
    Ekin_keV_amu.name = 'Ekin / amu'

    def Ekin_keV_qm(self):
        return self.Ekin_MeV() * self._charge_e() / self._mass_u()
    Ekin_keV_qm.unit = 'keV*q/m'
    Ekin_keV_qm.name = 'Ekin * q/m'

    def angle_xy(self):
        return np.arctan2(self._Py(), self._Px())
    angle_xy.unit = 'rad'
    angle_xy.name = 'anglexy'

    def angle_yz(self):
        return np.arctan2(self._Pz(), self._Py())
    angle_yz.unit = 'rad'
    angle_yz.name = 'angleyz'

    def angle_zx(self):
        return np.arctan2(self._Px(), self._Pz())
    angle_zx.unit = 'rad'
    angle_zx.name = 'anglezx'

    def angle_yx(self):
        return np.arctan2(self._Px(), self._Py())
    angle_yx.unit = 'rad'
    angle_yx.name = 'angleyx'

    def angle_zy(self):
        return np.arctan2(self._Py(), self._Pz())
    angle_zy.unit = 'rad'
    angle_zy.name = 'anglezy'

    def angle_xz(self):
        return np.arctan2(self._Pz(), self._Px())
    angle_xz.unit = 'rad'
    angle_xz.name = 'anglexz'

    def angle_xaxis(self):
        return np.arctan2(np.sqrt(self._Py()**2 + self._Pz()**2), self.Px())
    angle_xaxis.unit = 'rad'
    angle_xaxis.name = 'angle_xaxis'

    # ---- Functions to create a Histogram. ---

    def createHistgram1d(self, scalarfx, optargsh={'bins': 300},
                         simextent=False, simgrid=False, rangex=None,
                         weights=lambda x: 1):
        if simgrid:
            simextent = True
        xdata = scalarfx(self)
        # In case there are no particles
        if len(xdata) == 0:
            return [], []
        if rangex is None:
            rangex = [np.min(xdata), np.max(xdata)]
        if simextent:
            if hasattr(scalarfx, 'extent'):
                rangex = scalarfx.extent
        if simgrid:
            if hasattr(scalarfx, 'gridpoints'):
                optargsh['bins'] = scalarfx.gridpoints
        w = self.weight() * weights(self)
        h, edges = np.histogram(xdata, weights=w,
                                range=rangex, **optargsh)
        h = h / np.diff(edges)  # to calculate particles per xunit.
        return h, edges

    def createHistgram2d(self, scalarfx, scalarfy,
                         optargsh={'bins': [500, 500]}, simextent=False,
                         simgrid=False, rangex=None, rangey=None,
                         weights=lambda x: 1):
        """
        Creates an 2d Histogram.

        Attributes
        ----------
        scalarfx : function
            returns a list of scalar values for the x axis.
        scalarfy : function
            returns a list of scalar values for the y axis.
        simgrid : boolean, optional
            enforces the same grid as used in the simulation.
            Implies simextent=True. Defaults to False.
        simextent : boolean, optional
            enforces, that the axis show the same extent as used in the
            simulation. Defaults to False.
        weights : function, optional
            applies additional weights to the macroparticles, for example
            "ParticleAnalyzer.Ekin_MeV"".
            Defaults to "lambda x:1".
        """
        if simgrid:
            simextent = True
        xdata = scalarfx(self)
        ydata = scalarfy(self)
        if len(xdata) == 0:
            return [[]], [0, 1], [1]
        # TODO: Falls rangex oder rangy gegeben ist,
        # ist die Gesamtteilchenzahl falsch berechnet, weil die Teilchen die
        # ausserhalb des sichtbaren Bereiches liegen mitgezaehlt werden.
        if rangex is None:
            rangex = [np.min(xdata), np.max(xdata)]
        if rangey is None:
            rangey = [np.min(ydata), np.max(ydata)]
        if simextent:
            if hasattr(scalarfx, 'extent'):
                rangex = scalarfx.extent
            if hasattr(scalarfy, 'extent'):
                rangey = scalarfy.extent
        if simgrid:
            if hasattr(scalarfx, 'gridpoints'):
                optargsh['bins'][0] = scalarfx.gridpoints
            if hasattr(scalarfy, 'gridpoints'):
                optargsh['bins'][1] = scalarfy.gridpoints
        w = self.weight() * weights(self)  # Particle Size * additional weights
        h, xedges, yedges = np.histogram2d(xdata, ydata,
                                           weights=w, range=[rangex, rangey],
                                           **optargsh)
        h = h / (xedges[1] - xedges[0]) / (yedges[1] - yedges[0])
        return h, xedges, yedges

    def createHistgramField1d(self, scalarfx, name='distfn', title=None,
                              **kwargs):
        """
        Creates an 1d Histogram enclosed in a Field object.

        Attributes
        ----------
        scalarfx : function
            returns a list of scalar values for the x axis.
        name : string, optional
            addes a name. usually used for generating savenames.
            Defaults to "distfn".
        title: string, options
            overrides the title. Autocreated if title==None.
            Defaults to None.
        **kwargs
            given to createHistgram1d.
        """
        if 'weights' in kwargs:
            name = kwargs['weights'].name
        h, edges = self.createHistgram1d(scalarfx, **kwargs)
        ret = Field(h, edges)
        ret.axes[0].grid_node = edges
        ret.name = name + ' ' + self.species
        ret.label = self.species
        if title:
            ret.name = title
        if hasattr(scalarfx, 'unit'):
            ret.axes[0].unit = scalarfx.unit
        if hasattr(scalarfx, 'name'):
            ret.axes[0].name = scalarfx.name
        ret.infos = self.getcompresslog()['all']
        ret.infostring = self.npart
        return ret

    def createHistgramField2d(self, scalarfx, scalarfy, name='distfn',
                              title=None, **kwargs):
        """
        Creates an 2d Histogram enclosed in a Field object.

        Attributes
        ----------
        scalarfx : function
            returns a list of scalar values for the x axis.
        scalarfy : function
            returns a list of scalar values for the y axis.
        name : string, optional
            addes a name. usually used for generating savenames.
            Defaults to "distfn".
        title: string, options
            overrides the title. Autocreated if title==None.
            Defaults to None.
        **kwargs
            given to createHistgram2d.
        """
        if 'weights' in kwargs:
            name = kwargs['weights'].name
        h, xedges, yedges = self.createHistgram2d(scalarfx, scalarfy, **kwargs)
        ret = Field(h, xedges, yedges)
        ret.axes[0].grid_node = xedges
        ret.axes[1].grid_node = yedges
        ret.name = name + self.species
        ret.label = self.species
        if title:
            ret.name = title
        ret.axes[0].unit = scalarfx.unit
        ret.axes[0].name = scalarfx.name
        ret.axes[1].unit = scalarfy.unit
        ret.axes[1].name = scalarfy.name
        ret.infostring = '{:.0f} npart in {:.0f} species'.format(self.npart, self.nspecies)
        ret.infos = self.getcompresslog()['all']
        return ret

    def createField(self, *scalarf, **kwargs):
        """
        Creates an n-d Histogram enclosed in a Field object.
        Try using this function first.

        Attributes
        ----------
        *args
            list of scalarfunctions that should be used for the axis.
            the number of args given determins the dimensionality of the
            field returned by this function.
        **kwargs
            given to createHistgram1d or createHistgram2d.
        """
        if self.simdimensions is None:
            return None
        if len(scalarf) == 1:
            return self.createHistgramField1d(*scalarf, **kwargs)
        elif len(scalarf) == 2:
            return self.createHistgramField2d(*scalarf, **kwargs)
        else:
            raise Exception('only 1d or 2d field creation implemented yet.')


