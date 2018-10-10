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
# Stephan Kuschel 2014-2018
"""
Particle related routines.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import copy
import warnings
from ..helper import PhysicalConstants as pc
import scipy.constants
from ._routines import SpeciesIdentifier, histogramdd
from ..helper import deprecated, append_doc_of
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
    return:
    1) a list (one value for each particle)
    2) a single scalar value if this property is equal for the entire
    species (as usual for 'mass' or 'charge').
    3) raise a KeyError on request if the property wasnt dumped.
    Once initiated, all implemented methods leave the object unchanged.
    A new instance is returned if needed.
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
        self.compresslog = []
        self._compressboollist = None
        self._cache = {}

        # create a method for every _atomicprops item.
        def makefunc(_self, key):
            def ret(_self):
                return _self._readatomic(key)
            return ret
        for key in self._atomicprops:
            setattr(_SingleSpecies, key, makefunc(self, key))

    def __copy__(self):
        '''
        returns a shallow copy of the object.
        This method is called by `copy.copy(obj)`.
        '''
        cls = type(self)
        ret = cls.__new__(cls)
        ret.__dict__.update(self.__dict__)
        # the content of _cache will be updated in the compress function,
        # But the copy needs its own dictionary
        for k in ['_cache', '_compressboollist', 'compresslog']:
            ret.__dict__[k] = copy.copy(self.__dict__[k])
        return ret

    @property
    def dumpreader(self):
        return self._dumpreader

    def __repr__(self):
        n = len(self)
        i = self.initial_npart()
        if n == i:
            s = '<SingleSpecies "{}" ({}) from "{}">'
            return s.format(self.species, n, self.dumpreader)
        else:
            s = '<SingleSpecies "{}" ({}/{} - {:.2%}) from "{}">'
            return s.format(self.species, n, i, n/i, self.dumpreader)

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
        like :meth:`compress`, but takes a ScalarProperty object instead which is required
        to evalute to a boolean list. Example:

        >>> ms2 = ms.filter('gamma > 12')

        Parameters
        ----------
        condition: str
          A string, which can also be used at `ms(condition)` and evaluates
        '''
        cond = self(condition)
        if name is None:
            name = condition.expr if condition.name is None else condition.name
        return self.compress(cond, name=name)

    def compress(self, condition, name='unknown condition'):
        """
        works like numpy.compress.
        Additionaly you can specify a name, that gets saved in the compresslog.
        Returns a new MultiSpecies instance. `compress` gives you a lot of control,
        but in most cases :meth:`filter` will be sufficient and keeps your code more readable.

        Parameters
        ----------
        condition: 1D numpy array
          Condition can be one out of two choices:

          * `condition =  [True, False, True, True, ... , True, False]`

            condition is a list of length N, specifing which particles to keep. The length
            of the list must be equal to the length of the MultiSpecies instance,
            otherwise a ValueError is raised.
            Example:

            >>> cfintospectrometer = lambda ms: ms('abs(angle_xaxis) < 30e-3')
            >>> ms2 = ms.compress(cfintospectrometer(ms), name='< 30mrad offaxis')

            Consider using :meth:`filter` instead.

          * `condtition = [7, 2000, 4, 5, 91, ... , 765, 809]`

            condition can be a list of arbitraty length. Only the particles
            with the ids listed here will be kept.

        name: str, optional
          an optional name of the condition. This can later be reviewed by
          calling 'self.compresslog()'
        """
        condition = np.asarray(condition)
        if condition.dtype is np.dtype('bool'):
            # Case 1:
            # condition is list of boolean values specifying particles to use
            ret = self._compress_bool(condition)
        else:
            # Case 2:
            # condition is list of particle IDs to use
            ret = self._compress_int(condition)
        ret.compresslog = np.append(self.compresslog, name)
        return ret

    def _compress_bool(self, condition):
        if not len(self) == len(condition):
            raise ValueError('number of particles ({:7n}) has to match'
                             'length of condition ({:7n})'
                             ''.format(len(self), len(condition)))
        ret = copy.copy(self)
        if ret._compressboollist is None:
            ret._compressboollist = condition
        else:
            ret._compressboollist[ret._compressboollist] = condition
        for key in ret._cache:
            if ret._cache[key].shape is not ():
                ret._cache[key] = ret._cache[key][condition]
        return ret

    def _compress_int(self, condition):
        condition = np.asarray(condition, dtype='int')
        # same as
        # bools = np.array([idx in condition for idx in self.ID()])
        # but benchmarked to be 1500 times faster :)
        condition.sort()
        ids = self.id()
        idx = np.searchsorted(condition, ids)
        idx[idx == len(condition)] = 0
        bools = condition[idx] == ids
        return self._compress_bool(bools)

    def uncompress(self):
        """
        Discard all previous runs of 'compress'
        """
        return type(self)(self.dumpreader, self.species)

    def __invert__(self):
        '''
        inverts which particles have been taken and which have not.
        '''
        ret = copy.copy(self)
        ret._cache = {}  # clear cache
        if self._compressboollist is None:
            ret._compressboollist = np.asarray(False)
        elif self._compressboollist.shape is () and bool(self._compressboollist) is False:
            ret._compressboollist = None
        else:
            ret._compressboollist = ~self._compressboollist
        ret.compresslog = np.append(self.compresslog, 'inverted')
        return ret

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

    def initial_npart(self):
        '''
        return the original number of particles.
        '''
        if self._compressboollist is None:
            return len(self)
        else:
            return len(self._compressboollist)

    # --- The Interface for particle properties using __call__ ---

    def _eval_single_sp(self, sp, _vars=None):
        # sp MUST be ScalarProperty
        # this docsting is forwared to __call__
        '''
        Variable resolution order
          1. try to find the value as a atomic particle property.
          2. try to find the value as a defined particle property in ``particle_scalars``.
          3. if not found look for an equally named attribute in ``scipy.constants``.
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

    def __copy__(self):
        '''
        returns a shallow copy of the object.
        This method is called by `copy.copy(obj)`.
        '''
        cls = type(self)
        ret = cls.__new__(cls)
        ret.__dict__.update(self.__dict__)
        ret._ssas = copy.copy(self._ssas)
        ret._compresslog = copy.copy(self._compresslog)
        return ret

    def __repr__(self):
        n = len(self)
        i = self.initial_npart
        if n == i:
            s = '<MultiSpecies including all "{:}" ({:})>'
            return s.format(self.species, n)
        else:
            s = '<MultiSpecies including "{:}" ({}/{} - {:.2%})>'
            return s.format(self.species, n, i, n/i)

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
    def initial_npart(self):
        '''
        Original number of particles (before the use of compression or filter).
        '''
        ret = 0
        for ssa in self._ssas:
            ret += ssa.initial_npart()
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
        This function modifies the current Object and always returns None.

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
        '''
        Operator overloading for the ``+`` operator.

        :returns: A new MultiSpecies object containing the combined collection
          of particles.

        .. note:: Particles present in ``self`` and ``other`` will be present twice in the
          returned object!
        '''
        ret = copy.copy(self)
        ret += other
        return ret

    def __iadd__(self, other):  # self += other
        '''
        Operator overloading for the ``+=`` operator.
        Adding MultiSpecies should give the feeling as if you were adding
        their particle lists. That is why there is no append function.
        Compare those outputs (numpy.array handles that differently!):

        >>> a=[1,2,3]; a.append([4,5]); print a
        >>> [1,2,3,[4,5]]
        >>> a=[1,2,3]; a += [4,5]; print a
        >>> [1,2,3,4,5]

        This function modifies the current object. Same behaviour as

        >>> a = [1,2]; b = a; b+=[7,8]; print(a,b)
        >>> [1, 2, 7, 8] [1, 2, 7, 8]

        .. seealso:: :meth:`__add__`
        '''
        for ssa in other._ssas:
            self._ssas.append(copy.copy(ssa))
        return self

    # --- compress related functions ---

    @append_doc_of(_SingleSpecies.filter)
    def filter(self, condition, name=None):
        if isinstance(condition, ScalarProperty):
            sp = condition
        else:
            sp = particle_scalars(condition)
        ret = copy.copy(self)
        ret._ssas = [ssa.filter(sp, name=name) for ssa in self._ssas]
        ret._compresslog = np.append(self._compresslog, str(condition))
        return ret

    @append_doc_of(_SingleSpecies.compress)
    def compress(self, condition, name='unknown condition'):
        condition = np.asarray(condition)
        i = 0
        ret = copy.copy(self)
        for ssai, ssa in enumerate(self._ssas):  # condition is list of booleans
            if condition.dtype == np.dtype('bool'):
                n = len(ssa)
                ret._ssas[ssai] = ssa.compress(condition[i:i + n], name=name)
                i += n
            else:  # condition is list of particle IDs
                ret._ssas[ssai] = ssa.compress(condition, name=name)
        ret._compresslog = np.append(self._compresslog, name)
        return ret

    def __invert__(self):
        '''
        invert the selection of particles.
        '''
        ret = copy.copy(self)
        ret._ssas = [~ssa for ssa in self._ssas]
        ret._compresslog = np.append(self._compresslog, 'inverted')
        return ret

    # --- other user friendly functions

    def compressfn(self, conditionf, name='unknown condition'):
        '''
        like :meth:`compress`, but accepts a function.

        Returns a new MultiSpecies instance.
        '''
        if hasattr(conditionf, 'name'):
            name = conditionf.name
        return self.compress(conditionf(self), name=name)

    def uncompress(self):
        '''
        Returns a new MultiSpecies instance, with all previous calls of
        :meth:`compress` or :meth:`filter` undone.
        '''
        ret = copy.copy(self)
        ret._compresslog = []
        ret._ssas = [s.uncompress() for s in self._ssas]
        return ret

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

        Supported types
          * ScalarProperty
          * str: will be converted to a ScalarProperty by particle_scalars.__call__.
            Therefore known quantities will be recognized
          * callable, which acts on the MultiSpecies object. This will work, but
            maybe removed in a future release.

        The list of known particle scalars can be accessed by
        ``postpic.particle_scalars``.

        Examples
          * ``self('x')``
          * ``self('sqrt(px**2 + py**2 + pz**2)')``
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
    X_um.unit = r'$\mu m$'
    X_um.name = 'X'

    @deprecated('Use self("y") instead.')
    def Y(self):
        return self('y')
    Y.unit = 'm'
    Y.name = 'Y'

    @deprecated('Use self("Y_mu") instead.')
    def Y_um(self):
        return self('y_um')
    Y_um.unit = r'$\mu m$'
    Y_um.name = 'Y'

    @deprecated('Use self("z") instead.')
    def Z(self):
        return self('z')
    Z.unit = 'm'
    Z.name = 'Z'

    @deprecated('Use self("z_um") instead.')
    def Z_um(self):
        return self('z_um')
    Z_um.unit = r'$\mu m$'
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

    def mean(self, expr, weights=None):
        '''
        The mean of a value given by the expression `expr`.
        The particle weight of the individual particles
        will be automatically included in the calculation.
        An additional weight can be given using the keyword `weights`.
        '''
        weights = '1' if weights is None else weights
        w = self('weight * ({})'.format(weights))
        return np.average(self(expr), weights=w)

    def var(self, expr, weights='1'):
        '''
        The variance of a value given by the expression `expr`.
        The particle weight of the individual particles
        will be automatically included in the calculation.
        An additional weight can be given using the keyword `weights`.
        '''
        w = self('weight * ({})'.format(weights))
        data = self(expr)
        m = np.average(data, weights=w)
        return np.average((data - m)**2, weights=w)

    def quantile(self, expr, q, weights=None):
        '''
        The qth-quantile of the distribution of a value given by the expression `expr`.
        q can be a scalar or a list of quantiles to be calculated simultaneously.
        The particle weight of the individual particles
        will be automatically included in the calculation.
        An additional weight can be given using the keyword `weights`.
        '''
        weights = '1' if weights is None else weights
        q = np.asarray(q)
        if np.any(q < 0) or np.any(q > 1):
            raise ValueError('Quantile(s) q ({:}) must be in range [0, 1]'.format(q))
        w = self('weight * ({})'.format(weights))
        data = self(expr)
        sortidx = np.argsort(data)
        wcs = np.cumsum(w[sortidx])
        idx = np.searchsorted(wcs, wcs[-1]*q)
        return data[sortidx[idx]]

    def median(self, expr, weights=None):
        '''
        The median of a value given by the expression `expr`.
        The particle weight of the individual particles
        will be automatically included in the calculation.
        An additional weight can be given using the keyword `weights`.
        '''
        return self.quantile(expr, 0.5, weights=weights)

    # ---- Functions to create a Histogram. ---

    def _createHistgram(self, *sps, **kwargs):
        """
        Creates an 3d Histogram.

        Attributes
        ----------
        *sps : a kind, that self.__call__ can evalute to
            returns a list of scalar values for the x/y/z axis.
        simgrid : boolean, optional
            enforces the same grid as used in the simulation.
            Implies simextent=True. Defaults to False.
        simextent : boolean, optional
            enforces, that the axis show the same extent as used in the
            simulation. Defaults to False.
        weights : function, optional
            applies additional weights to the macroparticles, for example
            'gamma' or 'q' to weight the particle by its charge.
            Defaults to '1' (no additional weight).
        rangex : list of two values, optional
            the xrange to include into the histogram
            Defaults to None, determins the range by the range of scalars given.
        rangey : list of two values, optional
            the yrange to include into the histogram
            Defaults to None, determins the range by the range of scalars given.
        rangez : list of two values, optional
            the zrange to include into the histogram
            Defaults to None, determins the range by the range of scalars given.
        """
        if 'optargsh' in kwargs:
            warnings.warn('keyword "optargsh" is deprecated. Use "bins" and "shape" '
                          'arguments directly on "createField".', category=DeprecationWarning)
            optargsh = kwargs.pop('optargsh')
            kwargs.update(optargsh)
        simextent = kwargs.pop('simextent', False)
        simgrid = kwargs.pop('simgrid', False)
        rangex = kwargs.pop('rangex', None)
        rangey = kwargs.pop('rangey', None)
        rangez = kwargs.pop('rangez', None)
        weights = kwargs.pop('weights', '1')
        force = kwargs.pop('force', False)
        bins = kwargs.pop('bins', None)
        shape = kwargs.pop('shape', None)
        if len(kwargs) > 0:
            raise TypeError("got an unexpected keyword argument {}'".format(kwargs))

        if len(sps) > 3:
            raise TypeError('Only 1D, 2D or 3D Histograms can be created.')

        if simgrid:
            simextent = True
        if force:
            try:
                data = [self(sp) for sp in sps]
            except (KeyError):
                data = [[]]  # Return empty histogram
        else:
            data = [self(sp) for sp in sps]
        # TODO: Falls rangex oder rangey gegeben ist,
        # ist die Gesamtteilchenzahl falsch berechnet, weil die Teilchen die
        # ausserhalb des sichtbaren Bereiches liegen mitgezaehlt werden.
        ranges = [rangex, rangey, rangez]
        if simextent:
            for i, sp in enumerate(sps):
                tmp = self.simextent(getattr(sp, 'symbol', sp))
                ranges[i] = tmp if tmp is not None else ranges[i]
        if simgrid:
            for i, sp in enumerate(sps):
                tmp = self.simgridpoints(getattr(sp, 'symbol', sp))
                if tmp is not None:
                    bins[i] = tmp
        if len(data[0]) == 0:  # no data points. create empy histogram
            h = np.zeros(bins)

            def createedges(rangei, n):
                if rangei is not None:
                    return np.linspace(rangei[0], rangei[1], n + 1)
                else:
                    return np.linspace(0, 1, n + 1)

            edges = [createedges(r, bins[i]) for i, r in zip(range(len(h)), ranges)]
            return h, edges  # empty histogram: h == 0 everywhere

        w = self('weight * ({})'.format(weights))  # Particle Size * additional weights
        h, edges = histogramdd(data,
                               weights=w, range=ranges,
                               bins=bins, shape=shape)
        dV = np.prod([edge[1] - edge[0] for edge in edges])
        h /= dV
        return h, edges  # h, (xedges, yedges, zedges)

    def createField(self, *sps, **kwargs):
        """
        Creates an n-d Histogram enclosed in a Field object.

        Parameters
        ----------
        *sps
            list of scalarfunctions/strings/scalar-properties,
            that will be evaluated to data for each axis.
            the number of args given determins the dimensionality of the
            field returned by this function (maximum 3)
        name: string, optional
            addes a name. usually used for generating savenames.
            Defaults to "distfn".
        title: string, options
            overrides the title. Autocreated if title==None.
            Defaults to None.
        simgrid : boolean, optional
            enforces the same grid as used in the simulation.
            Implies simextent=True. Defaults to False.
        simextent : boolean, optional
            enforces, that the axis show the same extent as used in the
            simulation. Defaults to False.
        weights : function, optional
            applies additional weights to the macroparticles, for example
            'gamma' or 'q' to weight the particle by its charge.
            Defaults to '1' (no additional weight).
        rangex : list of two values, optional
            the xrange to include into the histogram.
            Defaults to None, determins the range by the range of scalars given.
        rangey : list of two values, optional
            the yrange to include into the histogram.
            Defaults to None, determins the range by the range of scalars given.
        rangez : list of two values, optional
            the zrange to include into the histogram.
            Defaults to None, determins the range by the range of scalars given.
        bins: sequence or int
            The number of bins to use for each dimension
        shape: int
            possible choices are:
            * 0 - use nearest grid point (NGP)
            * 1 - use tophat shape of width 1 bin
            * 2 - triangular shape (default)
            * 3 - spline 3 shape
        """
        name = kwargs.pop('name', 'distfn')
        title = kwargs.pop('title', None)

        h, edges = self._createHistgram(*sps, **kwargs)
        edgekwargs = {name: edg for name, edg in zip(['xedges', 'yedges', 'zedges'], edges)}
        ret = Field(h, **edgekwargs)

        if 'weights' in kwargs:
            name = _findscalarattr(kwargs['weights'], 'name')
        ret.name = name + self.species
        ret.label = self.species
        ret.name = title if title else ret.name  # override if title is given
        for i, sp in enumerate(sps):
            ret.axes[i].unit = _findscalarattr(sp, 'unit')
            ret.axes[i].name = _findscalarattr(sp, 'name')
        ret.infostring = '{:.0f} npart in {:.0f} species'.format(self.npart, self.nspecies)
        ret.infos = self.getcompresslog()['all']
        return ret


class ParticleHistory(object):
    '''
    Represents a list of particles including their history that can be found in
    all the dumps defined
    by the simulation reader sr.

    Parameters
    ----------
    sr: iterable of datareader
        a collection of datareader to use. Usually a Simulationreader object
    speciess: string or iterable of strings
        a species name or a list of species names. Those particles can be included
        into the history.
    ids: iterable of int
        list of ids to use (default: None). If this is None all particles in speciess will
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

    def __copy__(self):
        '''
        returns a shallow copy of the object.
        This method is called by `copy.copy(obj)`.
        '''
        cls = type(self)
        ret = cls.__new__(cls)
        ret.__dict__.update(self.__dict__)
        # _updatelookupdict creates a new _id2i dictionary. Therefore no need to copy that here.
        return ret

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
        ms = ms.compress(self.ids)
        scalars = np.zeros((len(scalarfs), len(ms)))
        for i in range(len(scalarfs)):
            scalars[i, :] = ms(scalarfs[i])
        ids = np.asarray(ms('id'), dtype=np.int)
        del ms  # close file to not exceed limit of max open files
        return ids, scalars

    def skip(self, n):
        '''
        takes only everth (n+1)-th particle
        '''
        ret = copy.copy(self)
        ret.ids = self.ids[::n+1]
        ret._updatelookupdict()
        return ret

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
