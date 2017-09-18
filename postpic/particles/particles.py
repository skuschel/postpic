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
# Stephan Kuschel 2014-2017
"""
Particle related routines.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import copy
import warnings
from ..helper import PhysicalConstants as pc
import scipy.constants
from ..helper import SpeciesIdentifier, histogramdd, append_doc_of
from ..helper import deprecated
from ..datahandling import *
from .scalarproperties import ScalarProperty, ScalarPropertyContext, createdefaultscalarcontext

identifyspecies = SpeciesIdentifier.identifyspecies

# this file
__all__ = ['MultiSpecies', 'ParticleHistory', 'particle_scalars']
# imported
__all__ += ['identifyspecies']


particle_scalars = createdefaultscalarcontext()


def _findscalarattr(scalarf, attrib, default='unknown'):
    '''
    Tries to find the scalarf's attribute attrib like name or unit.
    returns None if not found
    '''
    ret = None
    if hasattr(scalarf, attrib):
        # scalarf is function or ScalarProperty
        ret = getattr(scalarf, attrib)
    if scalarf in particle_scalars:
        # scalarf is a string
        if hasattr(particle_scalars[scalarf], attrib):
            ret = getattr(particle_scalars[scalarf], attrib)
    return default if ret is None else ret


class _SingleSpecies(object):
    """
    used by the MultiSpecies class only.
    The _SingleSpecies will return atomic particle properties
    (see list below) as given by the dumpreader. Each property can thus
    return
    1) a list (one value for each particle)
    2) a siingle scalar value if this property is equal for the entire
    species (as usual for 'mass' or 'charge').
    3) raise a KeyError on request if the property wasnt dumped.
    """
    # List of atomic particle properties. Those will be requested from the dumpreader
    # All other particle properties will be calculated from these.
    _atomicprops = ['weight', 'x', 'y', 'z', 'px', 'py', 'pz', 'mass', 'charge', 'id', 'time']
    _atomicprops_synonyms = {'w': 'weight', 'm': 'mass', 'q': 'charge', 't': 'time'}

    def __init__(self, dumpreader, species):
        if species not in dumpreader.listSpecies():
            # A better way would be to test if len(self) == 0,
            # but that may require heavy IO
            raise(KeyError('species "{:}" does not exist in {:}'.format(species, dumpreader)))
        self.species = species
        self._dumpreader = dumpreader
        self.uncompress()
        # Variables will be read and added to self._cache when needed.

        # create a method for every _atomicprops item.
        def makefunc(_self, key):
            def ret(_self):
                return _self._readatomic(key)
            return ret
        for key in self._atomicprops:
            setattr(_SingleSpecies, key, makefunc(self, key))

    @property
    def dumpreader(self):
        return self._dumpreader

    def __str__(self):
        return '<_SingleSpecies ' + str(self.species) \
            + ' at ' + str(self._dumpreader) \
            + '(' + str(len(self)) + ')>'

    def _readatomic(self, key):
        '''
        Reads an atomic property, thus one out of
        weight, x, y, z, px, py, pz, mass, charge, ID
        (self._atomicprops)
        '''
        if key in self._cache:
            return self._cache[key]
        # if not cached, try to to find it
        if key in ['time']:
            ret = self._dumpreader.time()
        elif key in ['mass', 'charge']:
            try:
                ret = self._dumpreader.getSpecies(self.species, key)
            except(KeyError):
                # in the special case of mass or charge try to deduce mass or charge
                # from the species name.
                self._idfy = identifyspecies(self.species)
                ret = self._idfy[key]
        else:
            ret = self._dumpreader.getSpecies(self.species, key)
        # now that we have got the data, check if compress was used and/or maybe cache value
        ret = np.int64(ret) if key == 'id' else np.float64(ret)
        if ret.shape is ():  # cache single scalars always
            self._cache[key] = ret
        elif self._compressboollist is not None:
            ret = ret[self._compressboollist]  # avoid executing this line too often.
            self._cache[key] = ret
            # if memomry is low, caching could be skipped entirely.
            # See commit message for benchmark.
        return ret

    def filter(self, condition, name=None):
        '''
        like compress, but takes a ScalarProperty object instead which is required
        to evalute to a boolean list.
        '''
        cond = self(condition)
        if name is None:
            name = condition.expr if condition.name is None else condition.name
        self.compress(cond, name=name)

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
        condition = np.asarray(condition)
        if condition.dtype is np.dtype('bool'):
            # Case 1:
            # condition is list of boolean values specifying particles to use
            if not len(self) == len(condition):
                raise ValueError('number of particles ({:7n}) has to match'
                                 'length of condition ({:7n})'
                                 ''.format(len(self), len(condition)))
            if self._compressboollist is None:
                self._compressboollist = condition
            else:
                self._compressboollist[self._compressboollist] = condition
            for key in self._cache:
                if self._cache[key].shape is not ():
                    self._cache[key] = self._cache[key][condition]
            self.compresslog = np.append(self.compresslog, name)
        else:
            # Case 2:
            # condition is list of particle IDs to use
            condition = np.asarray(condition, dtype='int')
            # same as
            # bools = np.array([idx in condition for idx in self.ID()])
            # but benchmarked to be 1500 times faster :)
            condition.sort()
            ids = self.id()
            idx = np.searchsorted(condition, ids)
            idx[idx == len(condition)] = 0
            bools = condition[idx] == ids
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
        # return 0 if no valid dataset can be found
        ret = 0
        if self._compressboollist is not None:
            return np.count_nonzero(self._compressboollist)
        for key in self._atomicprops:
            try:
                # len(3) will yield a TypeError, len([3]) returns 1
                ret = len(self._readatomic(key))
                break
            except(TypeError, KeyError):
                pass
        return ret

    # --- The Interface for particle properties using __call__ ---

    def _eval_single_sp(self, sp, _vars=None):
        # sp MUST be ScalarProperty
        # this docsting is forwared to __call__
        '''
        Variable resolution order:
        --------------------------
        1. try to find the value as a atomic particle property.
        2. try to find the value as a defined particle property in `particle_scalars`.
        3. if not found look for an equally named attribute in `scipy.constants`.
        '''
        _vars = dict() if _vars is None else _vars
        expr = sp.expr
        for name in sp.input_names:
            # load each variable needed
            if name in _vars:
                # already loaded -> skip
                continue
            fullname = self._atomicprops_synonyms.get(name, name)
            if fullname in _vars:
                _vars[name] = _vars[fullname]
                continue
            if fullname in self._atomicprops:
                _vars[name] = getattr(self, fullname)()
                continue
            if name in particle_scalars:  # the public list of scalar values
                _vars[name] = self._eval_single_sp(particle_scalars[name], _vars=_vars)
                continue
            for source in [np, scipy.constants]:
                try:
                    _vars[name] = getattr(source, name)
                except(AttributeError):
                    pass
            if name not in _vars:
                raise KeyError('"{}" not found!'.format(name))
        return sp.evaluate(_vars)

    @append_doc_of(_eval_single_sp)
    def __call__(self, sp, _vars=None):
        if not isinstance(sp, ScalarProperty):
            raise TypeError('Argument must be a ScalarProperty object')
        return self._eval_single_sp(sp, _vars=_vars)


class MultiSpecies(object):
    """
    The MultiSpecies class. Different MultiSpecies can be
    added together to create a combined collection.

    **kwargs
    --------
    ignore_missing_species = False
        set to true to ignore missing species.

    The MultiSpecies class will return a list of values for every
    particle property.
    """

    def __init__(self, dumpreader, *speciess, **kwargs):
        # create 'empty' MultiSpecies
        self._ssas = []
        self._species = None  # trivial name if set
        self._compresslog = []
        # add particle species one by one
        for s in speciess:
            self.add(dumpreader, s, **kwargs)

    def __str__(self):
        return '<MultiSpecies including ' + str(self.species) \
            + '(' + str(len(self)) + ')>'

    @property
    def dumpreader(self):
        '''
        returns the dumpreader if the dumpreader of **all** species
        are pointing to the same dump. This should be mostly the case.

        Otherwise returns None.
        '''
        try:
            dr0 = self._ssas[0].dumpreader
            if all([dr0 == ssa.dumpreader for ssa in self._ssas]):
                return dr0
        except(IndexError, KeyError):
            return None

    def simextent(self, axis):
        '''
        the combined simextent for all species and dumps included in this MultiSpecies object.
        '''
        extents = np.asarray([ssa.dumpreader.simextent(axis) for ssa in self._ssas])
        mins = np.min(extents, axis=0)[::2]
        maxs = np.max(extents, axis=0)[1::2]
        return np.asarray([mins, maxs]).T.flatten()

    def simgridpoints(self, axis):
        '''
        this function is for convenience only and is likely to be removed in the future.
        Particlarly it is impossible to define the grid of the simulation if the
        MultiSpecies object consists of multiple dumps from different simulations.
        '''
        try:
            ret = self._ssas[0].dumpreader.simgridpoints(axis)
        except(AttributeError, KeyError):
            return None

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
        '''
        Number of species.
        '''
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
            # return trivial name if set
            return self._species
        return ' '.join(set(self.speciess))

    @species.setter
    def species(self, species):
        self._species = species

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

    def add(self, dumpreader, species, ignore_missing_species=False):
        '''
        adds a species to this MultiSpecies.

        Attributes
        ----------
        species can be a single species name
                or a reserved name for collection of species, such as
                ions    adds all available particles that are ions
                nonions adds all available particles that are not ions
                ejected
                noejected
                all

        Optional arguments
        --------
        ignore_missing_species = False
            set to True to ignore if the species is missing.
        '''
        keys = {'_ions': lambda s: identifyspecies(s)['ision'],
                '_nonions': lambda s: not identifyspecies(s)['ision'],
                '_ejected': lambda s: identifyspecies(s)['ejected'],
                '_noejected': lambda s: not identifyspecies(s)['ejected'],
                '_all': lambda s: True}
        if species in keys:
            ls = dumpreader.listSpecies()
            toadd = [s for s in ls if keys[species](s)]
            for s in toadd:
                self.add(dumpreader, s)
        else:
            if ignore_missing_species:
                try:
                    self._ssas.append(_SingleSpecies(dumpreader, species))
                except(KeyError):
                    pass
            else:
                self._ssas.append(_SingleSpecies(dumpreader, species))
        return

    # --- Operator overloading

    def __add__(self, other):  # self + other
        ret = copy.copy(self)
        ret += other
        return ret

    def __iadd__(self, other):  # self += other
        '''
        adding MultiSpecies should give the feeling as if you were adding
        their particle lists. Thats why there is no append function.
        Compare those outputs (numpy.array handles that differently!):
        a=[1,2,3]; a.append([4,5]); print a
        [1,2,3,[4,5]]
        a=[1,2,3]; a += [4,5]; print a
        [1,2,3,4,5]
        '''
        for ssa in other._ssas:
            self._ssas.append(copy.copy(ssa))
        return self

    # --- compress related functions ---

    def filter(self, condition, name=None):
        '''
        like compress, but takes a ScalarProperty or a str, which are required to
        evaluate to a boolean list to filter particles. This is the preferred method to
        filter particles by a value of their property.
        '''
        if isinstance(condition, ScalarProperty):
            sp = condition
        else:
            sp = particle_scalars(condition)
        for ssa in self._ssas:
            ssa.filter(sp, name=name)
        self._compresslog = np.append(self._compresslog, str(condition))

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
        condition = [1, 2, 4, 5, 9, ... , 805, 809]
        condition can be a list of arbitraty length, so only the particles
        with the ids listed here are kept.

        **kwargs
        --------
        name -- name the condition. This can later be reviewed by calling 'self.compresslog()'
        """
        condition = np.asarray(condition)
        i = 0
        for ssa in self._ssas:  # condition is list of booleans
            if condition.dtype == np.dtype('bool'):
                n = len(ssa)
                ssa.compress(condition[i:i + n], name=name)
                i += n
            else:  # condition is list of particle IDs
                ssa.compress(condition, name=name)
        self._compresslog = np.append(self._compresslog, name)

    # --- user friendly functions

    def compressfn(self, conditionf, name='unknown condition'):
        '''
        like "compress", but accepts a function.

        **kwargs
        --------
        name -- name the condition.
        '''
        if hasattr(conditionf, 'name'):
            name = conditionf.name
        self.compress(conditionf(self), name=name)

    def uncompress(self):
        '''
        undo all previous calls of "compress".
        '''
        self._compresslog = []
        for s in self._ssas:
            s.uncompress()

    def getcompresslog(self):
        ret = {'all': self._compresslog}
        for ssa in self._ssas:
            ret.update({ssa.species: ssa.compresslog})
        return ret

    # --- Methods to access particle properties

    @append_doc_of(_SingleSpecies.__call__)
    def __call__(self, expr):
        '''
        Access to particle properties via the expression,
        which is used to calculate them.

        This is **only** function to actually access the data. Every other function
        which allows data access must call this one internally!

        Supported Types:
        ----------------
        * ScalarProperty
        * str: will be converted to a ScalarProperty by particle_scalars.__call__.
               Therefore known quantities will be recognized
        * callable, that acts on the MultiSpecies object. This will work, but
          maybe removed in a future release.

        Examples
        --------
        self('x')
        self('sqrt(px**2 + py**2 + pz**2)')
        '''
        if isinstance(expr, ScalarProperty):
            # best case
            return self.__call_sp(expr)
        try:
            bs = basestring  # python2
        except(NameError):
            bs = str  # python3
        if isinstance(expr, bs):
            # create temporary ScalarProperty object
            sp = particle_scalars(expr)
            return self.__call_sp(sp)
        else:
            return self.__call_func(expr)

    def __call_sp(self, sp):
        # sp MUST be ScalarProperty
        def ssdata(ss):
            a = ss(sp)
            if a.shape is ():
                a = np.repeat(a, len(ss))
            return a
        if len(self._ssas) == 0:
            # Happens, if only missing species were added with
            # ignore_missing_species = True.
            return np.array([])
        data = (ssdata(ss) for ss in self._ssas)
        return np.hstack(data)

    def __call_func(self, func):
        # hope it does what it should...
        s = '''
        You are accessing particle properties via the function {}.
        The Calculation of particle properties using functions is deprecated.
        Use a str or a ScalarProperty object instead. This will also allow postic to enable
        certain optimizations. When in doubt, use the str.
        '''.format(str(func))
        warnings.warn(s, category=DeprecationWarning)
        return func(self)

    # --- "A scalar for every particle"-functions.

    @deprecated('Use self("{name}") instead.')
    def time(self):
        return self('time')
    time.name = 'time'
    time.unit = 's'

    @deprecated('Use self("{name}") instead.')
    def weight(self):
        return self('weight')
    weight.name = 'Particle weight'
    weight.unit = 'npartpermacro'

    @deprecated('Use self("id") instead.')
    def ID(self):
        return self('id')

    @deprecated('Use self("{name}") instead.')
    def mass(self):  # SI
        return self('mass')
    mass.unit = 'kg'
    mass.name = 'm'

    @deprecated('Use self("{name}") instead.')
    def mass_u(self):
        return self('mass_u')
    mass_u.unit = 'u'
    mass_u.name = 'm'

    @deprecated('Use self("{name}") instead.')
    def charge(self):  # SI
        return self('charge')
    charge.unit = 'C'
    charge.name = 'q'

    @deprecated('Use self("{name}") instead.')
    def charge_e(self):
        return self('charge_e')
    charge.unit = 'qe'
    charge.name = 'q'

    @deprecated('Use self("{name}") instead.')
    def Eruhe(self):
        return self('Eruhe')

    @deprecated('Use self("px") instead.')
    def Px(self):
        return self('px')
    Px.unit = ''
    Px.name = 'Px'

    @deprecated('Use self("py") instead.')
    def Py(self):
        return self('py')
    Py.unit = ''
    Py.name = 'Py'

    @deprecated('Use self("pz") instead.')
    def Pz(self):
        return self('pz')
    Pz.unit = ''
    Pz.name = 'Pz'

    @deprecated('Use self("p") instead.')
    def P(self):
        return self('p')
    P.unit = ''
    P.name = 'P'

    @deprecated('Use self("x") instead.')
    def X(self):
        return self('x')
    X.unit = 'm'
    X.name = 'X'

    @deprecated('Use self("x_um") instead.')
    def X_um(self):
        return self('x_um')
    X_um.unit = '$\mu m$'
    X_um.name = 'X'

    @deprecated('Use self("y") instead.')
    def Y(self):
        return self('y')
    Y.unit = 'm'
    Y.name = 'Y'

    @deprecated('Use self("Y_mu") instead.')
    def Y_um(self):
        return self('y_um')
    Y_um.unit = '$\mu m$'
    Y_um.name = 'Y'

    @deprecated('Use self("z") instead.')
    def Z(self):
        return self('z')
    Z.unit = 'm'
    Z.name = 'Z'

    @deprecated('Use self("z_um") instead.')
    def Z_um(self):
        return self('z_um')
    Z_um.unit = '$\mu m$'
    Z_um.name = 'Z'

    @deprecated('Use self("{name}") instead.')
    def beta(self):
        return self('beta')
    beta.unit = r'$\beta$'
    beta.name = 'beta'

    @deprecated('Use self("{name}") instead.')
    def betax(self):
        return self('betax')
    betax.unit = r'$\beta$'
    betax.name = 'betax'

    @deprecated('Use self("{name}") instead.')
    def betay(self):
        return self('betay')
    betay.unit = r'$\beta$'
    betay.name = 'betay'

    @deprecated('Use self("{name}") instead.')
    def betaz(self):
        return self('betaz')
    betaz.unit = r'$\beta$'
    betaz.name = 'betaz'

    @deprecated('Use self("v") instead.')
    def V(self):
        return self('v')
    V.unit = 'm/s'
    V.name = 'V'

    @deprecated('Use self("vx") instead.')
    def Vx(self):
        return self('vx')
    Vx.unit = 'm/s'
    Vx.name = 'Vx'

    @deprecated('Use self("vy") instead.')
    def Vy(self):
        return self('vy')
    Vy.unit = 'm/s'
    Vy.name = 'Vy'

    @deprecated('Use self("vz") instead.')
    def Vz(self):
        return self('vz')
    Vz.unit = 'm/s'
    Vz.name = 'Vz'

    @deprecated('Use self("{name}") instead.')
    def gamma(self):
        return self('gamma')
    gamma.unit = r'$\gamma$'
    gamma.name = 'gamma'

    @deprecated('Use self("{name}") instead.')
    def gamma_m1(self):
        return self('gamma_m1')
    gamma_m1.unit = r'$\gamma - 1$'
    gamma_m1.name = 'gamma_m1'

    @deprecated('Use self("{name}") instead.')
    def Ekin(self):
        return self('Ekin')
    Ekin.unit = 'J'
    Ekin.name = 'Ekin'

    @deprecated('Use self("{name}") instead.')
    def Ekin_MeV(self):
        return self('Ekin_MeV')
    Ekin_MeV.unit = 'MeV'
    Ekin_MeV.name = 'Ekin'

    @deprecated('Use self("{name}") instead.')
    def Ekin_MeV_amu(self):
        return self('Ekin_MeV_amu')
    Ekin_MeV_amu.unit = 'MeV / amu'
    Ekin_MeV_amu.name = 'Ekin / amu'

    @deprecated('Use self("{name}") instead.')
    def Ekin_MeV_qm(self):
        return self('Ekin_MeV_qm')
    Ekin_MeV_qm.unit = 'MeV*q/m'
    Ekin_MeV_qm.name = 'Ekin * q/m'

    @deprecated('Use self("{name}") instead.')
    def Ekin_keV(self):
        return self('Ekin_keV')
    Ekin_keV.unit = 'keV'
    Ekin_keV.name = 'Ekin'

    @deprecated('Use self("{name}") instead.')
    def Ekin_keV_amu(self):
        return self('Ekin_keV_amu')
    Ekin_keV_amu.unit = 'keV / amu'
    Ekin_keV_amu.name = 'Ekin / amu'

    @deprecated('Use self("{name}") instead.')
    def Ekin_keV_qm(self):
        return self('Ekin_keV_qm')
    Ekin_keV_qm.unit = 'keV*q/m'
    Ekin_keV_qm.name = 'Ekin * q/m'

    @deprecated('Use self("{name}") instead.')
    def angle_xy(self):
        return self('angle_xy')
    angle_xy.unit = 'rad'
    angle_xy.name = 'anglexy'

    @deprecated('Use self("{name}") instead.')
    def angle_yz(self):
        return self('ange_yz')
    angle_yz.unit = 'rad'
    angle_yz.name = 'angleyz'

    @deprecated('Use self("{name}") instead.')
    def angle_zx(self):
        return self('angle_zx')
    angle_zx.unit = 'rad'
    angle_zx.name = 'anglezx'

    @deprecated('Use self("{name}") instead.')
    def angle_yx(self):
        return self('angle_yx')
    angle_yx.unit = 'rad'
    angle_yx.name = 'angleyx'

    @deprecated('Use self("{name}") instead.')
    def angle_zy(self):
        return self('angle_zy')
    angle_zy.unit = 'rad'
    angle_zy.name = 'anglezy'

    @deprecated('Use self("{name}") instead.')
    def angle_xz(self):
        return self('angle_xz')
    angle_xz.unit = 'rad'
    angle_xz.name = 'anglexz'

    @deprecated('Use self("{name}") instead.')
    def angle_xaxis(self):
        return self('angle_xaxis')
    angle_xaxis.unit = 'rad'
    angle_xaxis.name = 'angle_xaxis'

    @deprecated('Use self("{name}") instead.')
    def r_xy(self):
        return self('r_xy')
    r_xy.unit = 'm'
    r_xy.name = 'r_xy'

    @deprecated('Use self("{name}") instead.')
    def r_yz(self):
        return self('r_yz')
    r_yz.unit = 'm'
    r_yz.name = 'r_yz'

    @deprecated('Use self("{name}") instead.')
    def r_zx(self):
        return self('r_zx')
    r_zx.unit = 'm'
    r_zx.name = 'r_zx'

    @deprecated('Use self("{name}") instead.')
    def r_xyz(self):
        return self('r_xyz')
    r_xyz.unit = 'm'
    r_xyz.name = 'r_xyz'
    # ---- Functions for measuring particle collection related values

    def mean(self, expr, weights='1'):
        '''
        the mean of a value given by the function func. The particle weight
        of the individual particles will be included in the calculation.
        An additional weight can be given as well.
        '''
        w = self('weight * ({})'.format(weights))
        return np.average(self(expr), weights=w)

    def var(self, expr, weights='1'):
        '''
        variance
        '''
        w = self('weight * ({})'.format(weights))
        data = self(expr)
        m = np.average(data, weights=w)
        return np.average((data - m)**2, weights=w)

    def quantile(self, expr, q, weights='1'):
        '''
        The qth-quantile of the distribution.
        '''
        if q < 0 or q > 1:
            raise ValueError('Quantile q ({:}) must be in range [0, 1]'.format(q))
        w = self('weight * ({})'.format(weights))
        data = self(expr)
        sortidx = np.argsort(data)
        wcs = np.cumsum(w[sortidx])
        idx = np.searchsorted(wcs, wcs[-1]*np.asarray(q))
        return data[sortidx[idx]]

    def median(self, expr, weights='1'):
        '''
        The median
        '''
        return self.quantile(expr, 0.5, weights=weights)

    # ---- Functions to create a Histogram. ---

    def _createHistgram1d(self, spx, optargsh={},
                          simextent=False, simgrid=False, rangex=None,
                          weights='1', force=False):
        '''
        creates a 1d histogram.
        spx must be of a kind, that self.__call__ can evalute to
        '''
        optargshdefs = {'bins': 300}
        optargshdefs.update(optargsh)
        optargsh = optargshdefs
        if simgrid:
            simextent = True
        if force:
            try:
                xdata = self(spx)
            except (KeyError):
                xdata = []  # Return empty histogram
        else:
            xdata = self(spx)
        if simextent:
            tmp = self.simextent(getattr(spx, 'symbol', spx))
            rangex = tmp if tmp is not None else rangex
        if simgrid:
            tmp = self.simgridpoints(getattr(spx, 'symbol', spx))
            if tmp is not None:
                optargsh['bins'] = tmp
        if len(xdata) == 0:
            h = np.zeros(optargsh['bins'])
            if rangex is not None:
                xedges = np.linspace(rangex[0], rangex[1], optargsh['bins'] + 1)
            else:
                xedges = np.linspace(0, 1, optargsh['bins'] + 1)
            return h, xedges  # empty histogram: h == 0 everywhere
        if rangex is None:
            rangex = [np.min(xdata), np.max(xdata)]
        w = self('weight * ({})'.format(weights))
        h, edges = histogramdd((xdata,), weights=w,
                               range=rangex, **optargsh)
        h = h / np.diff(edges)  # to calculate particles per xunit.
        return h, edges

    def _createHistgram2d(self, spx, spy,
                          optargsh={}, simextent=False,
                          simgrid=False, rangex=None, rangey=None,
                          weights='1', force=False):
        """
        Creates an 2d Histogram.

        Attributes
        ----------
        spx : a kind, that self.__call__ can evalute to
            returns a list of scalar values for the x axis.
        spy : a kind, that self.__call__ can evalute to
            returns a list of scalar values for the y axis.
        simgrid : boolean, optional
            enforces the same grid as used in the simulation.
            Implies simextent=True. Defaults to False.
        simextent : boolean, optional
            enforces, that the axis show the same extent as used in the
            simulation. Defaults to False.
        weights : function, optional
            applies additional weights to the macroparticles, for example
            "MultiSpecies.Ekin_MeV"".
            Defaults to "lambda x:1".
        """
        optargshdefs = {'bins': [500, 500]}
        optargshdefs.update(optargsh)
        optargsh = optargshdefs
        if simgrid:
            simextent = True
        if force:
            try:
                xdata = self(spx)
                ydata = self(spy)
            except (KeyError):
                xdata = []  # Return empty histogram
        else:
            xdata = self(spx)
            ydata = self(spy)
        # TODO: Falls rangex oder rangy gegeben ist,
        # ist die Gesamtteilchenzahl falsch berechnet, weil die Teilchen die
        # ausserhalb des sichtbaren Bereiches liegen mitgezaehlt werden.
        if simextent:
            tmp = self.simextent(getattr(spx, 'symbol', spx))
            rangex = tmp if tmp is not None else rangex
            tmp = self.simextent(getattr(spy, 'symbol', spy))
            rangey = tmp if tmp is not None else rangey
        if simgrid:
            for i, sp in enumerate([spx, spy]):
                tmp = self.simgridpoints(getattr(sp, 'symbol', sp))
                if tmp is not None:
                    optargsh['bins'][i] = tmp
        if len(xdata) == 0:
            h = np.zeros(optargsh['bins'])
            if rangex is not None:
                xedges = np.linspace(rangex[0], rangex[1], optargsh['bins'][0] + 1)
            else:
                xedges = np.linspace(0, 1, optargsh['bins'][0] + 1)
            if rangey is not None:
                yedges = np.linspace(rangey[0], rangey[1], optargsh['bins'][1] + 1)
            else:
                yedges = np.linspace(0, 1, optargsh['bins'][1] + 1)
            return h, xedges, yedges  # empty histogram: h == 0 everywhere
        if rangex is None:
            rangex = [np.min(xdata), np.max(xdata)]
        if rangey is None:
            rangey = [np.min(ydata), np.max(ydata)]
        w = self('weight * ({})'.format(weights))  # Particle Size * additional weights
        h, xedges, yedges = histogramdd((xdata, ydata),
                                        weights=w, range=[rangex, rangey],
                                        **optargsh)
        h = h / (xedges[1] - xedges[0]) / (yedges[1] - yedges[0])
        return h, xedges, yedges

    def _createHistgram3d(self, spx, spy, spz,
                          optargsh={}, simextent=False,
                          simgrid=False, rangex=None, rangey=None, rangez=None,
                          weights='1', force=False):
        """
        Creates an 3d Histogram.

        Attributes
        ----------
        spx : a kind, that self.__call__ can evalute to
            returns a list of scalar values for the x axis.
        spy : a kind, that self.__call__ can evalute to
            returns a list of scalar values for the y axis.
        spz : a kind, that self.__call__ can evalute to
            returns a list of scalar values for the z axis.
        simgrid : boolean, optional
            enforces the same grid as used in the simulation.
            Implies simextent=True. Defaults to False.
        simextent : boolean, optional
            enforces, that the axis show the same extent as used in the
            simulation. Defaults to False.
        weights : function, optional
            applies additional weights to the macroparticles, for example
            "MultiSpecies.Ekin_MeV"".
            Defaults to "lambda x:1".
        """
        optargshdefs = {'bins': [200, 200, 200]}
        optargshdefs.update(optargsh)
        optargsh = optargshdefs
        if simgrid:
            simextent = True
        if force:
            try:
                xdata = self(spx)
                ydata = self(spy)
                zdata = self(spz)
            except (KeyError):
                xdata = []  # Return empty histogram
        else:
            xdata = self(spx)
            ydata = self(spy)
            zdata = self(spz)
        # TODO: Falls rangex oder rangy gegeben ist,
        # ist die Gesamtteilchenzahl falsch berechnet, weil die Teilchen die
        # ausserhalb des sichtbaren Bereiches liegen mitgezaehlt werden.
        if simextent:
            tmp = self.simextent(getattr(spx, 'symbol', spx))
            rangex = tmp if tmp is not None else rangex
            tmp = self.simextent(getattr(spy, 'symbol', spy))
            rangey = tmp if tmp is not None else rangey
            tmp = self.simextent(getattr(spz, 'symbol', spz))
            rangez = tmp if tmp is not None else rangez
        if simgrid:
            for i, sp in enumerate([spx, spy, spz]):
                tmp = self.simgridpoints(getattr(sp, 'symbol', sp))
                if tmp is not None:
                    optargsh['bins'][i] = tmp
        if len(xdata) == 0:
            h = np.zeros(optargsh['bins'])
            if rangex is not None:
                xedges = np.linspace(rangex[0], rangex[1], optargsh['bins'][0] + 1)
            else:
                xedges = np.linspace(0, 1, optargsh['bins'][0] + 1)
            if rangey is not None:
                yedges = np.linspace(rangey[0], rangey[1], optargsh['bins'][1] + 1)
            else:
                yedges = np.linspace(0, 1, optargsh['bins'][1] + 1)
            if rangez is not None:
                zedges = np.linspace(rangez[0], rangez[1], optargsh['bins'][2] + 1)
            else:
                zedges = np.linspace(0, 1, optargsh['bins'][2] + 1)
            return h, xedges, yedges, zedges  # empty histogram: h == 0 everywhere
        if rangex is None:
            rangex = [np.min(xdata), np.max(xdata)]
        if rangey is None:
            rangey = [np.min(ydata), np.max(ydata)]
        if rangez is None:
            rangez = [np.min(zdata), np.max(zdata)]
        w = self('weight * ({})'.format(weights))  # Particle Size * additional weights
        h, xe, ye, ze = histogramdd((xdata, ydata, zdata),
                                    weights=w, range=[rangex, rangey, rangez],
                                    **optargsh)
        h = h / (xe[1] - xe[0]) / (ye[1] - ye[0]) / (ze[1] - ze[0])
        return h, xe, ye, ze

    def createHistgramField1d(self, spx, name='distfn', title=None,
                              **kwargs):
        """
        Creates an 1d Histogram enclosed in a Field object.

        Attributes
        ----------
        spx : str or ScalarProperty or function acting on a MultiSpecies object
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
            name = _findscalarattr(kwargs['weights'], 'name')
        h, edges = self._createHistgram1d(spx, **kwargs)
        ret = Field(h, edges)
        ret.axes[0].grid_node = edges
        ret.name = name + ' ' + self.species
        ret.label = self.species
        if title:
            ret.name = title
        ret.axes[0].unit = _findscalarattr(spx, 'unit')
        ret.axes[0].name = _findscalarattr(spx, 'name')
        ret.infos = self.getcompresslog()['all']
        ret.infostring = self.npart
        return ret

    def createHistgramField2d(self, spx, spy, name='distfn',
                              title=None, **kwargs):
        """
        Creates an 2d Histogram enclosed in a Field object.

        Attributes
        ----------
        spx : str or ScalarProperty or function acting on a MultiSpecies object
            returns a list of scalar values for the x axis.
        spy : str or ScalarProperty or function acting on a MultiSpecies object
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
            name = _findscalarattr(kwargs['weights'], 'name')
        h, xedges, yedges = self._createHistgram2d(spx, spy, **kwargs)
        ret = Field(h, xedges, yedges)
        ret.axes[0].grid_node = xedges
        ret.axes[1].grid_node = yedges
        ret.name = name + self.species
        ret.label = self.species
        if title:
            ret.name = title
        ret.axes[0].unit = _findscalarattr(spx, 'unit')
        ret.axes[0].name = _findscalarattr(spx, 'name')
        ret.axes[1].unit = _findscalarattr(spy, 'unit')
        ret.axes[1].name = _findscalarattr(spy, 'name')
        ret.infostring = '{:.0f} npart in {:.0f} species'.format(self.npart, self.nspecies)
        ret.infos = self.getcompresslog()['all']
        return ret

    def createHistgramField3d(self, spx, spy, spz, name='distfn',
                              title=None, **kwargs):
        """
        Creates an 3d Histogram enclosed in a Field object.

        Attributes
        ----------
        spx : str or ScalarProperty or function acting on a MultiSpecies object
            returns a list of scalar values for the x axis.
        spy : str or ScalarProperty or function acting on a MultiSpecies object
            returns a list of scalar values for the y axis.
        spz : str or ScalarProperty or function acting on a MultiSpecies object
            returns a list of scalar values for the z axis.
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
            name = _findscalarattr(kwargs['weights'], 'name')
        h, xedges, yedges, zedges = self._createHistgram3d(spx, spy, spz, **kwargs)
        ret = Field(h, xedges, yedges, zedges)
        ret.axes[0].grid_node = xedges
        ret.axes[1].grid_node = yedges
        ret.axes[2].grid_node = yedges
        ret.name = name + self.species
        ret.label = self.species
        if title:
            ret.name = title
        ret.axes[0].unit = _findscalarattr(spx, 'unit')
        ret.axes[0].name = _findscalarattr(spx, 'name')
        ret.axes[1].unit = _findscalarattr(spy, 'unit')
        ret.axes[1].name = _findscalarattr(spy, 'name')
        ret.axes[2].unit = _findscalarattr(spz, 'unit')
        ret.axes[2].name = _findscalarattr(spz, 'name')
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
        if len(scalarf) == 1:
            return self.createHistgramField1d(*scalarf, **kwargs)
        elif len(scalarf) == 2:
            return self.createHistgramField2d(*scalarf, **kwargs)
        elif len(scalarf) == 3:
            return self.createHistgramField3d(*scalarf, **kwargs)
        else:
            raise Exception('only 1d, 2d and 3d field creation implemented yet.')


class ParticleHistory(object):
    '''
    Represents a list of particles including their history that can be found in
    all the dumps defined
    by the simulation reader sr.

    Parameters
    ----------
    sr : a collection of datareader to use. Usually a Simulationreader object
    speciess : a species name or a list of species names. Those particles can be included
               into the history.
    ids : list of ids to use (default: None). If this is None all particles in speciess will
          be tracked. If a list of ids is given, these ids will be serached in speciess only.
    '''

    def __init__(self, sr, speciess, ids=None):
        # the simulation reader (collection of dumpreader)
        self.sr = sr
        # list of species names to search in for the particle id
        self.speciess = [speciess] if type(speciess) is str else speciess
        if ids is None:
            self.ids = self._findids()  # List of integers
        else:
            self.ids = np.asarray(ids, dtype=np.int)
        # lookup dict used by collect
        self._updatelookupdict()

    def _updatelookupdict(self):
        '''
        updates `self._id2i`.
        `self._id2i` is a dictionary mapping from a particle ID
        to the array index of that particle.
        '''
        self._id2i = {self.ids[i]: i for i in range(len(self.ids))}

    def _findids(self):
        '''
        finds which ids are prensent in all dumps and the speciess specified.
        '''
        idsfound = set()
        for dr in self.sr:
            ms = MultiSpecies(dr, *self.speciess, ignore_missing_species=True)
            idsfound |= set(ms('id'))
            del ms
        return np.asarray(list(idsfound), dtype=np.int)

    def __len__(self):
        # counts the number of particles present
        return len(self.ids)

    def _collectfromdump(self, dr, scalarfs):
        '''
        dr - the dumpreader
        scalarfs - a list of functions which return scalar values when applied to a dumpreader

        Returns:
           list of ids, [list scalar values, list of scalar values, ... ]
        '''
        ms = MultiSpecies(dr, *self.speciess, ignore_missing_species=True)
        ms.compress(self.ids)
        scalars = np.zeros((len(scalarfs), len(ms)))
        for i in range(len(scalarfs)):
            scalars[i, :] = ms(scalarfs[i])
        ids = ms('id')
        del ms  # close file to not exceed limit of max open files
        return ids, scalars

    def skip(self, n):
        '''
        takes only everth (n+1)-th particle
        '''
        self.ids = self.ids[::n+1]
        self._updatelookupdict()

    def collect(self, *scalarfs):
        '''
        Collects the given particle properties for all particles for all times.

        Parameters:
        -----------
        *scalarfs: the scalarfunction(s) defining the particle property

        Returns:
        --------
        numpy.ndarray holding the different particles in the same order as the list of `self.ids`,
        meaning the particle on position `particle_idx` has the ID `self.ids[particle_idx]`.
        every array element holds the history for a single particle.
        Indexorder of returned array: [particle_idx][scalarf_idx, collection_idx]
        '''
        particlelist = [list() for _ in range(len(self.ids))]
        for dr in self.sr:
            ids, scalars = self._collectfromdump(dr, scalarfs)
            for k in range(len(ids)):
                i = self._id2i[ids[k]]
                particlelist[i].append(scalars[:, k])
        ret = [np.asarray(p).T for p in particlelist]
        return ret
