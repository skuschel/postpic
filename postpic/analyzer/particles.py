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
        self.dumpreader = dumpreader
        self._idfy = identifyspecies(species)
        self._mass = self._idfy['mass']  # SI
        self._charge = self._idfy['charge']  # SI
        self.compresslog = []
        # Hold local copies to allow compress function
        # Any function will return None if property wasnt dumped.
        self._weightdata = dumpreader.getSpecies(species, 'weight')
        self._Xdata = dumpreader.getSpecies(species, 'x')
        self._Ydata = dumpreader.getSpecies(species, 'y')
        self._Zdata = dumpreader.getSpecies(species, 'z')
        self._Pxdata = dumpreader.getSpecies(species, 'px')
        self._Pydata = dumpreader.getSpecies(species, 'py')
        self._Pzdata = dumpreader.getSpecies(species, 'pz')
        self._ID = dumpreader.getSpecies(species, 'ID')

    @staticmethod
    def _compressifdumped(condition, data):
        if data is None:
            ret = None
        else:
            ret = np.compress(condition, data)
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
            assert self._weightdata.shape[0] == condition.shape[0], \
                'number of particles ({:7n}) has to match' \
                'length of condition ({:7n})' \
                ''.format(self._weightdata.shape[0], len(condition))
            self._weightdata = self._compressifdumped(condition, self._weightdata)
            self._Xdata = self._compressifdumped(condition, self._Xdata)
            self._Ydata = self._compressifdumped(condition, self._Ydata)
            self._Zdata = self._compressifdumped(condition, self._Zdata)
            self._Pxdata = self._compressifdumped(condition, self._Pxdata)
            self._Pydata = self._compressifdumped(condition, self._Pydata)
            self._Pzdata = self._compressifdumped(condition, self._Pzdata)
            self._ID = self._compressifdumped(condition, self._ID)
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
            return self.compress(bools, name=name)
        self.compresslog = np.append(self.compresslog, name)

    def uncompress(self):
        """
        Discard all previous runs of 'compress'
        """
        self.__init__(self.dumpreader, self.species)

    # --- Only very basic functions

    def __len__(self):  # = number of particles
        if self._weightdata is None:
            return 0
        else:
            return self.weight().shape[0]

# --- These functions are for practical use. Return None if not dumped.

    @staticmethod
    def _returnifdumped(data):
        if data is None:
            ret = None
        else:
            ret = np.float64(data)
        return ret

    def weight(self):  # np.float64(np.array([4.3])) == 4.3 may cause error
        return np.asfarray(self._weightdata, dtype='float64')

    def mass(self):  # SI
        return np.repeat(self._mass, self.weight().shape[0])

    def charge(self):  # SI
        return np.repeat(self._charge, self.weight().shape[0])

    def Px(self):
        return self._returnifdumped(self._Pxdata)

    def Py(self):
        return self._returnifdumped(self._Pydata)

    def Pz(self):
        return self._returnifdumped(self._Pzdata)

    def X(self):
        return self._returnifdumped(self._Xdata)

    def Y(self):
        return self._returnifdumped(self._Ydata)

    def Z(self):
        return self._returnifdumped(self._Zdata)

    def ID(self):
        if self._ID is None:
            ret = None
        else:
            ret = np.array(self._ID, dtype=int)
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
        self.X.__func__.extent = dumpreader.extent('x')
        self.X.__func__.gridpoints = dumpreader.gridpoints('x')
        self.X_um.__func__.extent = dumpreader.extent('x') * 1e6
        self.X_um.__func__.gridpoints = dumpreader.gridpoints('x')
        if self.simdimensions > 1:
            self.Y.__func__.extent = dumpreader.extent('y')
            self.Y.__func__.gridpoints = dumpreader.gridpoints('y')
            self.Y_um.__func__.extent = dumpreader.extent('y') * 1e6
            self.Y_um.__func__.gridpoints = dumpreader.gridpoints('y')
        if self.simdimensions > 2:
            self.Z.__func__.extent = dumpreader.extent('z')
            self.Z.__func__.gridpoints = dumpreader.gridpoints('z')
            self.Z_um.__func__.extent = self.dumpreader.extent('z') * 1e6
            self.Z_um.__func__.gridpoints = dumpreader.gridpoints('z')
        self.angle_xy.__func__.extent = np.real([-np.pi, np.pi])
        self.angle_yz.__func__.extent = np.real([-np.pi, np.pi])
        self.angle_zx.__func__.extent = np.real([-np.pi, np.pi])
        self.angle_offaxis.__func__.extent = np.real([0, np.pi])
        # add particle species one by one
        for s in speciess:
            self.add(dumpreader, s)

    def __str__(self):
        return '<ParticleAnalyzer including ' + str(self._speciess) \
            + '(' + str(len(self)) + ')>'

    @property
    def npart(self):
        '''
        Number of Particles.
        '''
        return self._weight().shape[0]

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

    def angle_offaxis(self):
        return np.arccos(self._Px() / (self.P() + 1e-300))
    angle_offaxis.unit = 'rad'
    angle_offaxis.name = 'angleoffaxis'

    # ---- Functions to create a Histogram. ---

    def createHistgram1d(self, scalarfx, optargsh={'bins': 300},
                         simextent=False, simgrid=False, rangex=None,
                         weights=lambda x: 1):
        if simgrid:
            simextent = True
        # In case there are no particles
        if len(scalarfx(self)) == 0:
            return [], []
        if rangex is None:
            rangex = [np.min(scalarfx(self)), np.max(scalarfx(self)) + 1e-7]
        if simextent:
            if hasattr(scalarfx, 'extent'):
                rangex = scalarfx.extent
        if simgrid:
            if hasattr(scalarfx, 'gridpoints'):
                optargsh['bins'] = scalarfx.gridpoints
        w = self.weight() * weights(self)
        h, edges = np.histogram(scalarfx(self), weights=w,
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
        if len(scalarfx(self)) == 0:
            return [], [], []
        # TODO: Falls rangex oder rangy gegeben ist,
        # ist die Gesamtteilchenzahl falsch berechnet, weil die Teilchen die
        # ausserhalb des sichtbaren Bereiches liegen mitgezaehlt werden.
        if rangex is None:
            rangex = [np.min(scalarfx(self)), np.max(scalarfx(self)) + 1e-7]
        if rangey is None:
            rangey = [np.min(scalarfy(self)), np.max(scalarfy(self)) + 1e-7]
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
        h, xedges, yedges = np.histogram2d(scalarfx(self), scalarfy(self),
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
        ret = Field(h)
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
        ret = Field(h)
        ret.axes[0].grid_node = xedges
        ret.axes[1].grid_node = yedges
        ret.name = name + self.species
        ret.label = self.species
        if title:
            ret.name = title
        ret.axes[0].unit = scalarfx.unit
        ret.axes[0].name = scalarfx.name
        ret.axes[1].unit = scalarfx.unit
        ret.axes[1].name = scalarfx.name
        ret.infostring = '{:.0f} part in {:.0f} species'.format(self.npart, self.nspecies)
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


