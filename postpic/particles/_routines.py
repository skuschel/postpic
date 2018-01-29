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
# Stephan Kuschel 2017
"""
Particle related functions.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import collections

from . import _particlestogrid as ptg
from ..helper import PhysicalConstants
import re

particleshapes = ptg.shapes

__all__ = ['histogramdd', 'SpeciesIdentifier']


def histogramdd(data, **kwargs):
    '''
    Creates a histogram of the data. This function has the similar signature and return values as
    `numpy.histogramdd`.
    In addition this function supports the `shape` keyword argument to choose
    the particle shape used. If used with `shape=0` the results of this function and the
    `numpy.histogramdd` are identical, however, this function is approx. factor 2 or 3 faster.

    Parameters
    ----------
    data: sequence of ndarray or ndarray (1D or 2D)
        The input (particle) data for the histogram.
         * A 1D numpy array (for 1D histogram).
         * A sequence providing the data for the different axis, i.e.
           `(datax, datay, dataz)` (preferred).
         * A (N, D)-array, i.e. `[[x1, y1, z1], [x2, y2, z2]]` -- must be a numpy array!
    bins: sequence or int
        The number of bins to use for each dimension
    range: sequence, optional
        A sequence of lower and upper bin edges to be used if the edges are not given
        explicitly in bins. Defaults to the minimum and maximum values along each dimension.
    weights: 1D numpy array
        The weights to be used for each data point
    shape: int
        possible choices are:
         * 0 - use nearest grid point (NGP)
         * 1 - use tophat shape of width 1 bin
         * 2 - triangular shape (default)
         * 3 - spline 3 shape

    Returns
    -------
    H : ndarray
        the final histogram
    edges : list
        A list of D arrays describing the edges for each dimension
    '''
    kwshape = kwargs.pop('shape', None)
    kwshape = 2 if kwshape is None else kwshape  # default value
    # default value need to be set separately, such that calling the function
    # with `shape=None` or the shape argument not given yields the same result.
    kwrange = kwargs.pop('range', None)
    kwweights = kwargs.pop('weights', None)
    kwbins = kwargs.pop('bins', None)
    if len(kwargs) > 0:
        raise TypeError("got an unexpected keyword argument {}'".format(kwargs))

    try:
        shape = data.shape
        if shape[0] > 3 and len(shape) == 2:  # (N, D) array
            # data[:,i] will create a view consuming only microseconds
            data = [data[:, i] for i in range(shape[1])]
    except(AttributeError):
        pass

    # upcast 1D if length 1 dimensions are omitted
    if np.isscalar(data[0]):  # [1,2,3]
        data = (data, )  # ([1,2,3],)
    if len(data) > 3:
        raise ValueError('Data with len {:} not supported. Maximum is 3D data.'.format(len(data)))
    if isinstance(kwrange, collections.Iterable) and np.isscalar(kwrange[0]):
        kwrange = (kwrange, )
    if np.isscalar(kwbins):
        kwbins = (kwbins, ) * len(data)
    if kwbins is None:
        binsdefs = {1: [800],
                    2: [500, 500],
                    3: [200, 200, 200]}
        kwbins = binsdefs[len(data)]

    # data is now (datax, datay, dataz)
    # make sure each is an ndarray. If it is already, this operation is fast.
    data = [np.asarray(d, dtype='float64') for d in data]
    if kwweights is not None:
        kwweights = np.asarray(kwweights, dtype='float64')
    # 1D, 2D, 3D
    ranges = [[None, None] for d in data]
    for ax, d in enumerate(data):
        for i, f in zip([0, 1], [np.min, np.max]):
            try:
                ranges[ax][i] = kwrange[ax][i]
                if ranges[ax][i] is None:
                    raise TypeError  # catch exception and fill value
                if not np.isscalar(ranges[ax][i]):
                    # if value can be accessed it must be a scalar value
                    raise ValueError('range="{}" not properly formatted.'.format(kwrange))
            except(TypeError):
                ranges[ax][i] = f(d)
    kwrange = ranges

    if len(data) == 1:  # ([1,2,3],)
        kwrange = kwrange[0]
        kwbins = kwbins[0]
        h, xedges = ptg.histogram(data[0],
                                  range=kwrange, bins=kwbins,
                                  weights=kwweights, shape=kwshape)
        return h, (xedges,)
    elif len(data) == 2:  # [[1,2,3], [4,5,6]]
        h, xedges, yedges = ptg.histogram2d(data[0],
                                            data[1],
                                            range=kwrange, bins=kwbins,
                                            weights=kwweights, shape=kwshape)
        return h, (xedges, yedges)
    elif len(data) == 3:  # [[1,2,3], [4,5,6], [7,8,9]]
        h, xe, ye, ze = ptg.histogram3d(data[0],
                                        data[1],
                                        data[2],
                                        range=kwrange, bins=kwbins,
                                        weights=kwweights, shape=kwshape)
        return h, (xe, ye, ze)
    else:
        assert False, 'Internal error'


class SpeciesIdentifier(PhysicalConstants):
    '''
    This Class provides static methods for deriving particle properties
    from species Names. The only reason for this to be a class is that it
    can be used as a mixin.
    '''

    def _specdict(mass, charge, ision):
        return dict(mass=mass * PhysicalConstants.me,
                    charge=charge * PhysicalConstants.qe,
                    ision=ision)

    # if in default, use this
    _defaults = {'electrongold': _specdict(1, -1, False),
                 'proton': _specdict(1836.2 * 1, 1, True),
                 'Proton': _specdict(1836.2 * 1, 1, True),
                 'ionp': _specdict(1836.2, 1, True),
                 'ion': _specdict(1836.2 * 12, 1, True),
                 'c6': _specdict(1836.2 * 12, 1, True),
                 'ionf': _specdict(1836.2 * 19, 1, True),
                 'Palladium': _specdict(1836.2 * 106, 0, True),
                 'Palladium1': _specdict(1836.2 * 106, 1, True),
                 'Palladium2': _specdict(1836.2 * 106, 2, True),
                 'Ion': _specdict(1836.2, 1, True),
                 'Photon': _specdict(0, 0, False),
                 'Positron': _specdict(1, 1, False),
                 'positron': _specdict(1, 1, False),
                 'gold1': _specdict(1836.2 * 197, 1, True),
                 'gold3': _specdict(1836.2 * 197, 3, True),
                 'gold4': _specdict(1836.2 * 197, 4, True),
                 'gold2': _specdict(1836.2 * 197, 2, True),
                 'gold7': _specdict(1836.2 * 197, 7, True),
                 'gold10': _specdict(1836.2 * 197, 10, True),
                 'gold20': _specdict(1836.2 * 197, 20, True)}

    #  unit: amu
    _masslistelement = {'H': 1, 'He': 4,
                        'Li': 6.9, 'C': 12, 'N': 14, 'O': 16, 'F': 19, 'Ne': 20.2,
                        'Na': 23, 'Al': 27, 'Si': 28, 'S': 32, 'Cl': 35.5, 'Ar': 40,
                        'Ti': 47.9, 'Cr': 52, 'Fe': 55.8, 'Cu': 63.5, 'Zn': 65.4, 'Kr': 83.8,
                        'Rb': 85.5, 'Zr': 91.2, 'Pd': 106.4, 'Ag': 107.8, 'Sn': 118.7,
                        'Xe': 131.3,
                        'W': 183.8, 'Pt': 195, 'Au': 197, 'Hg': 200.6, 'Pb': 207.2}

    @staticmethod
    def isejected(species):
        s = species.replace('/', '')
        r = re.match(r'(ejected_)(.*)', s)
        return r is not None

    @classmethod
    def ision(cls, species):
        return cls.identifyspecies(species)['ision']

    @classmethod
    def identifyspecies(cls, species):
        """
        Returns a dictionary containing particle informations deduced from
        the species name.
        The following keys in the dictionary will always be present:
        name   species name string
        mass    kg (SI)
        charge  C (SI)
        tracer  boolean
        ejected boolean

        Valid Examples:
        Periodic Table symbol + charge state: c6, F2, H1, C6b
        ionm#c# defining mass and charge:  ionm12c2, ionc20m110
        advanced examples:
        ejected_tracer_ionc5m20b, ejected_tracer_electronx,
        ejected_c6b, tracer_proton, protonb
        """
        ret = {'tracer': False, 'ejected': False, 'name': species}
        s = species.replace('/', '_')

        if s in cls._defaults:
            # if species name is found in cls._defaults use it and
            # return result.
            ret.update(cls._defaults[s])
            return ret

        # Regex for parsing ion species name.
        # See docsting for valid examples
        regex = r'(?P<prae>(.*_)*)' \
                r'((?P<elem>((?!El)[A-Z][a-z]?))(?P<elem_c>\d*)|' \
                r'(?P<ionmc>(ionc(?P<c1>\d+)m(?P<m2>\d+)|ionm(?P<m1>\d+)c(?P<c2>\d+)))' \
                r')?' \
                r'(?P<plus>(Plus)*)' \
                r'(?P<electron>[Ee]le[ck])?' \
                r'(?P<suffix>\w*?)$'
        r = re.match(regex, s)
        if r is None:
            raise Exception('Species ' + str(s) +
                            ' does not match regex name pattern: ' +
                            str(regex))
        regexdict = r.groupdict()
        # print(regexdict)

        # recognize anz prae and add dictionary key
        if regexdict['prae']:
            for i in regexdict['prae'].split('_'):
                key = i.replace('_', '')
                if not key == '':
                    ret[key] = True

        # Excluding patterns start here
        # 1) Name Element + charge state: C1, C6, F2, F9, Au20, Pb34a
        if regexdict['elem']:
            ret['mass'] = float(cls._masslistelement[regexdict['elem']]) * \
                1836.2 * cls.me
            if regexdict['elem_c'] == '':
                ret['charge'] = 0
            else:
                ret['charge'] = float(regexdict['elem_c']) * cls.qe
            ret['ision'] = True
        # 2) ionmc like
        elif regexdict['ionmc']:
            if regexdict['c1']:
                ret['mass'] = float(regexdict['m2']) * 1836.2 * cls.me
                ret['charge'] = float(regexdict['c1']) * cls.qe
                ret['ision'] = True

            if regexdict['c2']:
                ret['mass'] = float(regexdict['m1']) * 1836.2 * cls.me
                ret['charge'] = float(regexdict['c2']) * cls.qe
                ret['ision'] = True

        # charge may be given via plus
        if regexdict['plus']:
            charge = len(regexdict['plus'])/4 * cls.qe
            if ret['charge'] == 0:
                ret['charge'] = charge

        # Elektron can be appended to any ion name, so overwrite.
        if regexdict['electron']:
            ret['mass'] = cls.me
            ret['charge'] = -1 * cls.qe
            ret['ision'] = False

        if not (('mass' in ret) and ('charge' in ret) and ('ision' in ret)):
            raise Exception('species ' + species + ' not recognized.')

        return ret
